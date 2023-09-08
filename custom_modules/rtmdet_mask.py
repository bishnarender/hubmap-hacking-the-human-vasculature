# this file is a modified version of the original:
# https://github.com/tascj/kaggle-hubmap-hacking-the-human-vasculature/blob/main/custom_modules/rtmdet_mask.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align, batched_nms
from mmdet.registry import MODELS
from mmdet.models.detectors import RTMDet
from mmdet.models.utils import unpack_gt_instances
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import _do_paste_mask
from mmengine.structures import InstanceData

# RTMDet => Real-Time Object Detectors.
@MODELS.register_module()
class RTMDetWithMaskHead(RTMDet):

    def __init__(self, mask_head, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mask_head = MODELS.build(mask_head)
        assert self.train_cfg.mask_pos_mode in (
            # 'assigned_fpn_level',
            'all',
            'weighted_sum',
        )
        self.strides = [
            stride[0] for stride in self.bbox_head.prior_generator.strides
        ]
        if self.train_cfg.mask_pos_mode == 'weighted_sum':
            # .Parameter(data=None, requires_grad=True) => a kind of Tensor that is to be considered a module parameter.
            self.fpn_weight = nn.Parameter(
                torch.ones(len(self.strides), dtype=torch.float32),
                requires_grad=True,
            )
            self.fpn_weight_relu = nn.ReLU()
            self.eps = 1e-6

    def loss(self, batch_inputs, batch_data_samples):
        # batch_inputs.shape => torch.Size([2, 3, 768, 768])
        img_feats = self.extract_feat(batch_inputs)
        # type(img_feats) => tuple
        # img_feats[0].shape, img_feats[1].shape, img_feats[2].shape => torch.Size([2, 320, 96, 96]) torch.Size([2, 320, 48, 48]) torch.Size([2, 320, 24, 24])
        # img_feats[0] => ....grad_fn=<SiluBackward0>)
        # img_feats[1] => ....grad_fn=<SiluBackward0>)
        # img_feats[2] => ....grad_fn=<SiluBackward0>)
        
        # type(self.bbox_head) =>  <class 'mmdet.models.dense_heads.rtmdet_head.RTMDetSepBNHead'>
        
        # len(batch_data_samples) => 2
        # batch_data_samples[0].ignored_instances.bboxes.shape => torch.Size([0, 4])
        # batch_data_samples[0].ignored_instances.masks.shape => torch.Size([0, 56, 56])
        # batch_data_samples[0].ignored_instances.labels.shape => torch.Size([0])        
        # batch_data_samples[0].gt_instances.bboxes.shape => torch.Size([1, 4])
        # batch_data_samples[0].gt_instances.masks.shape => torch.Size([1, 56, 56])
        # batch_data_samples[0].gt_instances.labels.shape => torch.Size([1])

        # batch_data_samples[1].ignored_instances.bboxes.shape => torch.Size([0, 4])
        # batch_data_samples[1].ignored_instances.masks.shape => torch.Size([0, 56, 56])
        # batch_data_samples[1].ignored_instances.labels.shape => torch.Size([0])
        # batch_data_samples[1].gt_instances.bboxes.shape => torch.Size([2, 4])
        # batch_data_samples[1].gt_instances.masks.shape => torch.Size([2, 56, 56])
        # batch_data_samples[1].gt_instances.labels.shape => torch.Size([2])
        
        losses = self.bbox_head.loss(img_feats, batch_data_samples)
        # 'loss_cls': this key corresponds to the classification loss.
        # losses => {'loss_cls': [tensor(1.1772, device='cuda:0', grad_fn=<DivBackward0>), tensor(0.2935, device='cuda:0', grad_fn=<DivBackward0>), tensor(0.0014, device='cuda:0', grad_fn=<DivBackward0>)], 'loss_bbox': [tensor(0.6874, device='cuda:0', grad_fn=<DivBackward0>), tensor(0.2632, device='cuda:0', grad_fn=<DivBackward0>), tensor(0., device='cuda:0', grad_fn=<DivBackward0>)]}
        
        strides = self.strides
        mask_feat_size = self.train_cfg.mask_roi_size
        
        # len(batch_data_samples) => 2
        # type(unpack_gt_instances(batch_data_samples)), len(unpack_gt_instances(batch_data_samples)) => <class 'tuple'>, 3 
        batch_gt_instances = unpack_gt_instances(batch_data_samples)[0]
        # len(batch_gt_instances) => 2
        # batch_gt_instances[0].bboxes.shape => torch.Size([1, 4])
        # batch_gt_instances[1].bboxes.shape => torch.Size([2, 4])
        
        num_bboxes = sum(img_gt_instances.bboxes.size(0) for img_gt_instances in batch_gt_instances)
        #print(num_bboxes)
        
        if num_bboxes == 0:
            losses['loss_mask'] = self.mask_head(img_feats[-1]).sum() * 0
            return losses

        gt_masks = []
        gt_bboxes = []
        
        for img_gt_instances in batch_gt_instances:
            img_masks = img_gt_instances.masks
            img_bboxes = img_gt_instances.bboxes
            img_gt_roi_masks = img_masks  # assuming cropped in data pipeline
            gt_masks.append(img_gt_roi_masks)
            gt_bboxes.append(img_bboxes)
            
        gt_masks = torch.cat(gt_masks)
        # gt_masks.shape => torch.Size([3, 56, 56])
        
        mask_feats = []
        mask_gt = []
        
        # mask_feat_size => 28
        # strides => [8, 16, 32]  ( 768/96 =>8, 768/48 => 16, 768/24 => 32 )        
        for stride, feat in zip(strides, img_feats):
            # torchvision.ops.roi_align(input: Tensor, boxes: Union[Tensor, List[Tensor]], output_size: None, spatial_scale: float = 1.0, sampling_ratio: int = - 1, aligned: bool = False) => 
            #        performs Region of Interest (RoI) align operator with average pooling, as described in Mask R-CNN.
            #        extract fixed-size feature maps from different regions of an input feature map.
            # boxes => list of tensors where each tensor represents the boxes for a each batch element. 
            # sampling_ratio => number of sampling points in the interpolation grid used to compute the output value of each pooled output bin.
            
            roi_feats = roi_align(feat,
                                  gt_bboxes,
                                  mask_feat_size,
                                  
                                  # spatial_scale => e.g., 96x96 feature maps results from (1/8) scaling of the image 768x768.
                                  spatial_scale=1 / stride, 
                                  sampling_ratio=0,
                                  aligned=True)

            # roi_feats.shape => torch.Size([3, 320, 28, 28])              (same shape for all loop rounds)
            # 3 => num_bboxes
            mask_feats.append(roi_feats)
            mask_gt.append(gt_masks)

        if self.train_cfg.mask_pos_mode == 'all':
            mask_feats = torch.cat(mask_feats)
            mask_gt = torch.cat(mask_gt)
            
        else:  # weighted sum
            # self.fpn_weight.shape => torch.Size([3])
            weight = self.fpn_weight_relu(self.fpn_weight)
            weight = weight / (weight.sum() + self.eps)
            # weight.shape => torch.Size([3])            
            mask_feats = sum(mask_feat * w for mask_feat, w in zip(mask_feats, weight))
            # mask_feats.shape => torch.Size([3, 320, 28, 28])
            mask_gt = mask_gt[0]
            
        # type(self.mask_head) => <class 'mmdet.models.roi_heads.mask_heads.fcn_mask_head.FCNMaskHead'>
        
        # self.mask_head(mask_feats).shape => torch.Size([3, 1, 56, 56])
        mask_pred = self.mask_head(mask_feats).squeeze(1)
        # mask_pred.shape => torch.Size([3, 56, 56])
        
        # TODO: support multi-gpu avg
        losses['loss_mask'] = F.binary_cross_entropy_with_logits(mask_pred, mask_gt)

        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        # len(batch_inputs), len(batch_data_samples) => 1, 1
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]
        # batch_img_metas => [{'scale_factor': (1.5, 1.5), 'ori_shape': (512, 512), 'pad_shape': (768, 768), 'batch_input_shape': (768, 768), 'img_id': 0, 'img_path': 'data/train/06034408218a.tif', 'img_shape': (768, 768)}]

        hflip_tta = self.test_cfg.get('hflip_tta', False)
        if hflip_tta:
            # flip first so img_feats can be used for mask prediction
            img_feats = self.extract_feat(batch_inputs.flip([-1]))
            results_list_flip = self.bbox_head.predict(img_feats,
                                                batch_data_samples,
                                                rescale=False) # keep input scale
        
        img_feats = self.extract_feat(batch_inputs)
        # len(img_feats) => 3
        # img_feats[0].shape => torch.Size([1, 320, 96, 96])
        
        # batch_data_samples[0].gt_instances.bboxes.shape => torch.Size([8, 4])
        # batch_data_samples[0].gt_instances.labels.shape => torch.Size([8])
        results_list_orig = self.bbox_head.predict(img_feats, batch_data_samples, rescale=False)
        # len(results_list_orig) => 1
        # results_list_orig[0].bboxes.shape => torch.Size([300, 4])              (300 for all predictions)
        # results_list_orig[0].scores.shape => torch.Size([300])
        # results_list_orig[0].labels.shape => torch.Size([300])

        
        if hflip_tta:
            img_w = batch_inputs.size(-1)
            results_list = []
            for r1, r2 in zip(results_list_orig, results_list_flip):
                r2.bboxes[:, [0, 2]] = img_w - r2.bboxes[:, [2, 0]]  # inplace
                bboxes = torch.cat([
                    r1.bboxes, r2.bboxes
                ])
                scores = torch.cat([r1.scores, r2.scores])
                labels = torch.cat([r1.labels, r2.labels])
                keep = batched_nms(bboxes, scores, labels, iou_threshold=self.test_cfg.nms.iou_threshold)
                results_tta = InstanceData()
                results_tta.bboxes = bboxes[keep]
                results_tta.scores = scores[keep]
                results_tta.labels = labels[keep]
                results_list.append(results_tta)
        else:
            results_list = results_list_orig

        for img_idx, (results, img_meta) in enumerate(zip(results_list, batch_img_metas)):
            pred_bboxes = results.bboxes
            mask_feats = []
            for stride, feat in zip(self.strides, img_feats):
                roi_feats = roi_align(feat, [pred_bboxes],
                                      self.train_cfg.mask_roi_size,
                                      spatial_scale=1 / stride,
                                      sampling_ratio=0,
                                      aligned=True)
                # roi_feats.shape  => torch.Size([300, 320, 28, 28])               (same shape for all loop rounds)
                # 300 => num_bboxes 
                mask_feats.append(roi_feats)

            if self.train_cfg.mask_pos_mode == 'all':
                mask_feats = torch.cat(mask_feats)
            else:  # weighted sum
                weight = self.fpn_weight_relu(self.fpn_weight)
                weight = weight / (weight.sum() + self.eps)
                mask_feats = sum(mask_feat * w for mask_feat, w in zip(mask_feats, weight))
                
            # mask_feats.shape => torch.Size([300, 320, 28, 28])
            mask_pred = self.mask_head(mask_feats)
            # mask_pred.shape => torch.Size([300, 1, 56, 56])

            # rescale bboxes
            assert img_meta.get('scale_factor') is not None
            
            # .new_tensor(data, *, dtype=None, ...) => returns a new Tensor with data as the tensor data.
            # .repeat(*sizes) => repeats this tensor along the specified dimensions.
            #        sizes => the number of times to repeat this tensor along each dimension.

            # img_meta['scale_factor'] => (1.5, 1.5)
            # results.bboxes.new_tensor(img_meta['scale_factor']).shape => torch.Size([2])
            # results.bboxes.new_tensor(img_meta['scale_factor']).repeat((1, 2)).shape) => torch.Size([1, 4])
            results.bboxes /= results.bboxes.new_tensor(img_meta['scale_factor']).repeat((1, 2))

            img_h, img_w = img_meta['ori_shape'][:2]
            
            # TODO: chunking
            # TODO: paste on CPU
            # _do_paste_mask(masks, boxes, ..., skip_empty) => paste instance masks according to boxes.            
            #       masks => (Tensor): N, 1, H, W
            #       boxes => boxes (Tensor): N, 4
            #       skip_empty => (bool): False i.e., the whole image will be pasted. "_do_paste_mask" will return a mask of shape (N, img_h, img_w) and an empty tuple.
            
            # _do_paste_mask(mask_pred, pred_bboxes, img_h, img_w, False)[0].shape => torch.Size([300, 512, 512])
            img_mask = _do_paste_mask(mask_pred, pred_bboxes, img_h, img_w, False)[0] > 0  # score_thr=0.5
            
            results.masks = img_mask

        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)
        
        return batch_data_samples
