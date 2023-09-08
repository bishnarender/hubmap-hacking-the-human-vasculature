# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import BaseBoxes
from mmdet.utils import ConfigType
from mmdet.models.task_modules.assigners.assign_result import AssignResult

INF = 100000000
EPS = 1.0e-7


from mmdet.models.task_modules import DynamicSoftLabelAssigner


@TASK_UTILS.register_module()
class IgnoreMaskDynamicSoftLabelAssigner(DynamicSoftLabelAssigner):

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        """Assign gt to priors.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes``
                attribute data that is ignored during training and testing.
                Defaults to None.
        Returns:
            obj:`AssignResult`: The assigned result.
        """
        gt_bboxes = gt_instances.bboxes        
        gt_labels = gt_instances.labels
        # gt_bboxes.shape => torch.Size([4, 4])
        # gt_labels.shape => torch.Size([4])


        num_gt = gt_bboxes.size(0)

        decoded_bboxes = pred_instances.bboxes
        pred_scores = pred_instances.scores # cls
        priors = pred_instances.priors # anchors
        num_bboxes = decoded_bboxes.size(0)
        
        # decoded_bboxes.shape => torch.Size([12096, 4])
        # pred_scores.shape => torch.Size([12096, 3])
        # priors.shape => torch.Size([12096, 4])
        
        # .new_full(size, fill_value, *, dtype=None, ...) => returns a Tensor of "size" size "filled" with fill_value. 
        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes, ), 0, dtype=torch.long)
        
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                      -1,
                                                      dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        #/////////////////////////////////////////
        # code snippet calculates whether the top-left of "anchor boxes" is inside ground truth bounding boxes by calculating distances between their "top-left coordinates" and the corners of the ground truth boxes.
        
        prior_center = priors[:, :2]
        # prior_center.shape => torch.Size([12096, 2])
        if isinstance(gt_bboxes, BaseBoxes):
            is_in_gts = gt_bboxes.find_inside_points(prior_center)
        else:
            # prior_center[:, None].shape => torch.Size([12096, 1, 2])
            # gt_bboxes[:, :2].shape => torch.Size([4, 2])
            
            # Tensor boxes will be treated as horizontal boxes by defaults.
            lt_ = prior_center[:, None] - gt_bboxes[:, :2] 
            # lt_ => difference between the anchor box center (prior_center) and the top-left of the ground truth boxes.
            
            rb_ = gt_bboxes[:, 2:] - prior_center[:, None] 
            # rb_ => difference between the bottom-right of the ground truth boxes and anchor box center (prior_center).
            
            # lt_.shape, rb_.shape => torch.Size([12096, 4, 2]), torch.Size([12096, 4, 2])
            deltas = torch.cat([lt_, rb_], dim=-1)
            # deltas.shape => torch.Size([12096, 4, 4])
            
            # deltas.min(dim=-1).values.shape => torch.Size([12096, 4])
            is_in_gts = deltas.min(dim=-1).values > 0

        valid_mask = is_in_gts.sum(dim=1) > 0
        # valid_mask.shape => torch.Size([12096])
        
        #////////////////////////////////////
        
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)
        # valid_decoded_bbox.shape, num_valid => torch.Size([316, 4]), 316
        
        if num_valid == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                      -1,
                                                      dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        
        # MODIFIED
        # if hasattr(gt_instances, 'masks'):
        #     gt_center = center_of_mass(gt_instances.masks, eps=EPS)
        # elif isinstance(gt_bboxes, BaseBoxes):
        if isinstance(gt_bboxes, BaseBoxes):
            gt_center = gt_bboxes.centers
        else:
            # Tensor boxes will be treated as horizontal boxes by defaults
            gt_center = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2.0
            
        valid_prior = priors[valid_mask]
        strides = valid_prior[:, 2] # picking the x of bottom-right of anchor-boxes.
        # strides.shape => torch.Size([316])
                
        # valid_prior[:, None, :2]. shape, gt_center[None, :, :].shape, strides[:, None].shape, self.soft_center_radius => 
        # torch.Size([316, 1, 2]), torch.Size([1, 4, 2]), torch.Size([316, 1]), 3.0
        
        # ( valid_prior[:, None, :2] - gt_center[None, :, :] ).pow(2).sum(-1).sqrt().shape => torch.Size([316, 4])
        
        # Euclidean distance between top-left of each valid anchor box and the center of each ground truth bounding box. distance is normalized by the x of bottom-right of anchor-boxes.
        distance = ( valid_prior[:, None, :2] - gt_center[None, :, :] ).pow(2).sum(-1).sqrt() / strides[:, None]
        
        # .pow(input, exponent, *, out=None) => takes the power of each element in input with exponent and returns a tensor with the result.
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)
        # soft_center_prior.shape => torch.Size([316, 4])
        
        # type(self.iou_calculator) => <class 'mmdet.models.task_modules.assigners.iou2d_calculator.BboxOverlaps2D'>
        pairwise_ious = self.iou_calculator(valid_decoded_bbox, gt_bboxes)        
        # pairwise_ious.shape, self.iou_weight, EPS => torch.Size([316, 4]), 3.0, 1e-07
        # pairwise_ious => each row have "iou score" of each valid anchor box with all "gt bboxes".
        
        iou_cost = -torch.log(pairwise_ious + EPS) * self.iou_weight
        # iou_cost.shape => torch.Size([316, 4])
        
        # gt_labels.shape, pred_scores.shape, F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1]).shape =>
        # torch.Size([4]), torch.Size([12096, 3]), torch.Size([4, 3])
        
        # F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1]).float().unsqueeze(0) => torch.Size([1, 4, 3])
        gt_onehot_label = ( F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1]).float().unsqueeze(0).repeat(
                                                                                                    num_valid, 1, 1))
        # gt_onehot_label.shape => torch.Size([316, 4, 3])
        
        # gt_onehot_label[:3] => 
        # tensor([[[1., 0., 0.],
        #          [1., 0., 0.],
        #          [0., 0., 1.],
        #          [1., 0., 0.]],

        #         [[1., 0., 0.],
        #          [1., 0., 0.],
        #          [0., 0., 1.],
        #          [1., 0., 0.]],

        #         [[1., 0., 0.],
        #          [1., 0., 0.],
        #          [0., 0., 1.],
        #          [1., 0., 0.]]], device='cuda:0')        
        
        # valid_pred_scores.shape, valid_pred_scores.unsqueeze(1).shape => torch.Size([316, 3]), torch.Size([316, 1, 3])
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
        # valid_pred_scores.shape => torch.Size([316, 4, 3])
        
        # pairwise_ious[..., None].shape => torch.Size([316, 4, 1])
        soft_label = gt_onehot_label * pairwise_ious[..., None]
        # soft_label.shape => torch.Size([316, 4, 3])
        scale_factor = soft_label - valid_pred_scores.sigmoid()
        
        # F.binary_cross_entropy_with_logits(valid_pred_scores, soft_label, reduction='none').shape => torch.Size([316, 4, 3])
        soft_cls_cost = F.binary_cross_entropy_with_logits(valid_pred_scores, soft_label, reduction='none') * \
                                                                                scale_factor.abs().pow(2.0)
        # soft_cls_cost.shape => torch.Size([316, 4, 3])
        
        soft_cls_cost = soft_cls_cost.sum(dim=-1)
        # soft_cls_cost.shape => torch.Size([316, 4])

        cost_matrix = soft_cls_cost + iou_cost + soft_center_prior
        
        
        # torch.sum(valid_mask == True) => tensor(316, device='cuda:0')
        
        # type(self.dynamic_k_matching) => <class 'method'>
        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(cost_matrix, pairwise_ious, num_gt, valid_mask)
        # matched_pred_ious.shape, matched_gt_inds.shape => torch.Size([20]), torch.Size([20])
        # matched_gt_inds => 
        #        tensor([2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 2, 2, 2, 2, 2, 2], device='cuda:0')        
        #         i.e., indices of gt bboxes that are matched with top k bboxes present in "valid_decoded_bbox".
                
        # torch.sum(valid_mask == True) => tensor(20, device='cuda:0')
        
        # assigned_gt_inds[valid_mask].shape => torch.Size([20])       
        
        # convert to AssignResult format
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1 # 1 is added to differentiate indices with "other values" (which are zero).
        # assigned_gt_inds.shape => torch.Size([12096])
        
        # .new_full(size, fill_value, *, dtype=None, ...) => returns a Tensor of "size" size "filled" with fill_value. 
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1) 
        
        # gt_labels => tensor([0, 0, 2, 0], device='cuda:0')
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        # assigned_labels.shape => torch.Size([12096])
        
        max_overlaps = assigned_gt_inds.new_full((num_bboxes, ), -INF, dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        # max_overlaps.shape => torch.Size([12096])
        
        # num_gt => 4
        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
