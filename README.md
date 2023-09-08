## hubmap-hacking-the-human-vasculature
## score at 3rd position is achieved.
![hubmap-submission](https://github.com/bishnarender/hubmap-hacking-the-human-vasculature/assets/49610834/0c69c295-ef32-48e9-bdad-fce7f38c5831)

### Start 
-----
For better understanding of project, read the files in the following order:
1. eda.ipynb 
2. all_in_one.ipynb
3. hubmap-submission.ipynb

<b>Code has been explained in the associated files to the "all_in_one.ipynb".</b>

For the majority of instance segmentation models, the AP mainly depends on bounding box (bbox) prediction, while the precision of "mask prediction" has a minor impact. Therefore, when dealing with instance segmentation tasks, usually focus on optimizing bbox accuracy.  Mask prediction can be done by other models (like the mask head of Mask R-CNN or any semantic segmentation model). Here, during submission "mask prediction" has been handled by the Mask R-CNN for both.

Multiple EMA models has been utilised here in a single training run, along with a fixed learning rate. 

Models has been built, train and tested over <b>"mmdetection tool"</b>. Model components have 5 types.
* head: the component for specific tasks, e.g., bbox prediction and mask prediction.
* neck: the component between backbones and heads, e.g., FPN, PAFPN.
* backbone: usually an FCN (Fully Convolution Network) to extract feature maps, e.g., ResNet, MobileNet, CSPNeXt.
* roi extractor: the part for extracting RoI features from feature maps, e.g., RoI Align.
* loss: the component in head for calculating losses, e.g., FocalLoss, L1Loss, and GHMLoss.

### RTMDet Model
-----
![hubmap](https://github.com/bishnarender/hubmap-hacking-the-human-vasculature/assets/49610834/f1617cc3-aad7-4abc-95b9-80e89cc8ba2f)

<b>"mul" box</b> performs multiplication of input ([BS, 4, 96, 96]) with respective stride 8 (96 = 768/8).
<code>
#- https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/rtmdet_head.py
#- type(self.rtm_reg[idx]) => <class 'torch.nn.modules.conv.Conv2d'>
#- reg_feat.shape  => torch.Size([2, 320, 96, 96]) #- 2 = BS
#- stride[0] = 8
reg_dist = self.rtm_reg\[idx\](reg_feat).exp() * stride[0]
#- reg_dist => "regression distribution" represents the predicted "offsets or deltas" that are used to adjust the coordinates of bounding boxes during object detection.
</code>

<b>[BS, 3, 96, 96]</b> is a raw classification score at scale 1/8 (96 = 768/8). number of channels = num_anchors\*num_classes i.e., num_anchors = 1 and num_classes=3. 

<b>[BS, 4, 96, 96]</b> is predicted "offsets or deltas" at scale 1/8 (96 = 768/8). number of channels = num_anchors*4.

The term "regression" refers to the process of predicting continuous numerical values that need to be adjusted from some reference values. In object detection, regression is used to predict the adjustments (deltas) that need to be applied to the coordinates of the default or anchor bounding boxes to better fit the actual objects in the image.

Raw classification score at each scale is rearranged in the format [BS, 9216, 3], [BS, 2304, 3] and [BS, 576, 3]. Finally, concatenated to shape [BS, 12096, 3]. 
<code>
#- https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/rtmdet_head.py 
flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_score in cls_scores], 1)
</code>
This shaped score is used in calculation of "soft classification cost". Soft classification cost is a part of the "cost matrix". Cost matrix is used in the extraction of the top k matching bbox. Refer "class IgnoreMaskDynamicSoftLabelAssigner" in file "custom_modules/dynamic_soft_label_assigner.py".

Further, raw classification score is also used in calculation of QualityFocalLoss (i.e., our first target loss) as follows:
<code>
#- https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/rtmdet_head.py 
#- cls.shape => torch.Size([2, 3, 96, 96]) #- 2 = BS 
#- labels.shape => torch.Size([2, 9216])
#- assign_metrics.shape => torch.Size([2, 9216])
#- label_weights.shape => torch.Size([2, 9216])
cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels).contiguous()
labels = labels.reshape(-1)
assign_metrics = assign_metrics.reshape(-1)
label_weights = label_weights.reshape(-1)
targets = (labels, assign_metrics)

#- type(self.loss_cls) => <class 'mmdet.models.losses.gfocal_loss.QualityFocalLoss'>
loss_cls = self.loss_cls(cls_score, targets, label_weights, avg_factor=1.0)
</code>
assign_metrics have the values of corresponding "iou score". labels and assign_metrics have values only at indices for which anchor boxes have matching with ground truth bboxes. 

Let's see how "class QualityFocalLoss" calculated the "quality focal loss". First, binary_cross_entropy_with_logits is calculated between raw cls_score and zeros, multiplied by the "square of sigmoid of raw cls_score". Second, "scale_factor" is calculated from matching bboxes by the difference between iou_score and predicted probability of label. Third, again binary_cross_entropy_with_logits is calculated between iou_scores and raw prediction of labels, multiplied by the "square of absolute of scale_factor". The matching loss positions in first cross entropy loss are replaced with values obtained from second cross entropy loss.
<code>
#- https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/losses/gfocal_loss.py
#- pred => cls_score.
pred_sigmoid = pred.sigmoid()
scale_factor = pred_sigmoid
zerolabel = scale_factor.new_zeros(pred.shape)
loss = F.binary_cross_entropy_with_logits(pred, zerolabel, reduction='none') * scale_factor.pow(beta)

#- pred_sigmoid[pos, pos_label] => "pos" have indices at which "predicted bboxes" have match with "gt bboxes". "pos_label" have labels for "predicted bboxes" fetched from corresponding labels of matching gt bboxes.
scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
loss[pos, pos_label] = F.binary_cross_entropy_with_logits(pred[pos, pos_label], score[pos], reduction='none') * scale_factor.abs().pow(beta)
loss = loss.sum(dim=1, keepdim=False)
#- loss.shape => torch.Size([18432]) #- 18432 = 2*9216 #- 2 = BS 
</code> 
This is "quality focal loss" for the first stage i.e., at scale 1/8 and output (96,96). Loss is further normalized by the sum of iou scores.

Anchor Boxes are generated for each scale [1/8,1/16,1/32] using <b>torch.meshgrid(*tensors, indexing=None)</b>. For example, for the scale 1/8 and output (96,96) the anchor boxes are generated as follows:
<code>
#- https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/prior_generators/point_generator.py
#- MlvlPointGenerator
#- 8 = stride = 768/96.
shift_x = (torch.arange(0, 96, device=device) +  0) * 8 
shift_y = (torch.arange(0, 96, device=device) +  0) * 8
shift_yy, shift_xx = torch.meshgrid(shift_y, shift_x)
stride_w = shift_xx.new_full((shift_xx.shape[0], ), 8).to(dtype)
stride_h = shift_xx.new_full((shift_yy.shape[0], ), 8).to(dtype)
shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h], dim=-1)
#- shifts (i.e., our anchor boxes) =>
#- tensor([[ 0.,  0.,  8.,  8.],
#-         [ 8.,  0.,  8.,  8.],
#-         [16.,  0.,  8.,  8.],
#-         ...
#-         [752.,  0.,  8.,  8.],
#-         [760.,  0.,  8.,  8.]],             
#-         ...
#-         ...
#-         [ 0.,  760.,  8.,  8.],
#-         [ 8.,  760.,  8.,  8.],
#-         [16.,  760.,  8.,  8.],
#-         ...
#-         [760.,  760.,  8.,  8.]],             
#-         [760.,  760.,   8.,   8.]], device='cuda:0')                      

#- shifts.shape => torch.Size([9216, 4]) 
</code>

Predicted "offsets or deltas" at each scale is rearranged in the format [BS, 9216, 4], [BS, 2304, 4] and [BS, 576, 4]. These predicted "offsets" are subtracted from "respective coordinates" of "respective anchor boxes ([9216, 4], [2304, 4] and [576, 4])". 
<code>
#- https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/rtmdet_head.py 
bbox_pred = distance2bbox(anchor, bbox_pred)
#- https://github.com/open-mmlab/mmdetection/blob/main/mmdet/structures/bbox/transforms.py
#- points is our anchor boxes and distance is predicted offsets.
#- points[..., 0].shape, distance[..., 0].shape => torch.Size([9216]), torch.Size([2, 9216]) #- 2 = BS
x1 = points[..., 0] - distance[..., 0]
#- x1.shape => torch.Size([2, 9216])
y1 = points[..., 1] - distance[..., 1]
x2 = points[..., 0] + distance[..., 2]
y2 = points[..., 1] + distance[..., 3]

bboxes = torch.stack([x1, y1, x2, y2], -1)
#- bboxes.shape => torch.Size([2, 9216, 4]) #- 2 = BS
</code>
Finally, resultant anchor boxes are concatenated to shape [BS, 12096, 4].

From these anchor boxes top-k matching bbox are extracted using <b>iou score and many more</b>. 
<code>
#- https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/assigners/dynamic_soft_label_assigner.py
#- pairwise_ious => each row have "iou score" of each valid anchor box with all "gt bboxes".
topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
...
_, pos_idx = torch.topk( cost[:, gt_idx], k=dynamic_ks[gt_idx], largest=False)
...
</code>
Extraction process is explained along the code present in "class IgnoreMaskDynamicSoftLabelAssigner" in file "custom_modules/dynamic_soft_label_assigner.py".

Further, top-k matching anchor boxes and target bboxes are used in calculation of "generalized iou loss (GIoU loss)" (i.e., our second target loss (loss_bbox)). Count of target bboxes is changed as one target bbox have match with many anchor boxes. This is how GIoU loss calculated:
<code>
#- https://github.com/open-mmlab/mmdetection/blob/main/mmdet/structures/bbox/bbox_overlaps.py
#- bboxes1.shape, bboxes2.shape => torch.Size([68, 4]), torch.Size([68, 4])
#- bboxes1 => anchor boxes, bboxes2 => targets
#- area = (x2 - x1) * (y2 - y1)
area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * ( bboxes1[..., 3] - bboxes1[..., 1])
area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * ( bboxes2[..., 3] - bboxes2[..., 1])

#- compute the (max(x1,x2), max(y1,y2)) from the corresponding coordinates of bboxes1 (x1,y1) and bboxes2 (x2,y2).
#- effectively calculates the coordinates of the bottom-left corner of the intersection for each pair.
#- bottom-left because next we have "wh = rb-lt".
lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])

#- compute the (min(x1,x2), min(y1,y2)) from the corresponding coordinates of bboxes1 (x1,y1) and bboxes2 (x2,y2).
#- right-top.
rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])

#- non-negative width and height of the intersection region.
wh = (rb - lt).clamp(min=0)

#- area of intersection region.
overlap = wh[..., 0] * wh[..., 1]
union = area1 + area2 - overlap

eps = union.new_tensor([eps])
#- eps => tensor([1.0000e-06], device='cuda:0')
union = torch.max(union, eps)
ious = overlap / union

enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
#- area of the minimum enclosing bounding box that encompasses both corresponding input bounding boxes.
enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]

enclose_area = torch.max(enclose_area, eps)
gious = ious - (enclose_area - union) / enclose_area
loss_bbox = 1 - gious
</code>
Loss is further normalized by the "sum of weights of bboxes". GIoU is always a lower bound for IoU.

In no intersection case, IoU is 0 when two boxes are far no matter how far apart they are. But GIoU increases when two boxes approach each other.

Separate "mask head (FCNMaskHead)" is been employed in RTMDet model. And, mask loss (i.e., loss_mask) calculation is explained along the code in "method loss" of "class RTMDetWithMaskHead" in file "custom_modules/rtmdet_mask.py". Mask loss is calculated using roi_align function of "torchvision.ops" and mask head "class mmdet.models.roi_heads.mask_heads.fcn_mask_head.FCNMaskHead".

### Mask R-CNN Model
-----
![hubmap_](https://github.com/bishnarender/hubmap-hacking-the-human-vasculature/assets/49610834/be73004d-3960-4d15-8410-9f6d5f9c2235)
![hubmap_1_](https://github.com/bishnarender/hubmap-hacking-the-human-vasculature/assets/49610834/cce3df4c-e9c9-45be-8b38-5b6d34219b3d)

<b>[BS, 3, 320, 320], [BS, 3, 160, 160], ..., and [BS, 3, 20, 20]</b> are our raw "classification scores" or "classification predictions" at different scales [1/4, 1/8, 1/16, 1/32, 1/64].

Anchor Boxes are generated for each scale [1/4, 1/8, 1/16, 1/32, 1/64] using <b>torch.meshgrid(*tensors, indexing=None)</b>, with a slightly different method from RTMDet. For example, for the scale 1/4 and output (320,320) the anchor boxes are generated as follows:
<code>
#- https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/prior_generators/anchor_generator.py
#- AnchorGenerator
#- len(self.base_anchors) => 5
base_anchors = self.base_anchors[level_idx].to(device).to(dtype) #- level_idx = 0
#- base_anchors => 
#- tensor([[-22.6274, -11.3137,  22.6274,  11.3137],
#-         [-16.0000, -16.0000,  16.0000,  16.0000],
#-         [-11.3137, -22.6274,  11.3137,  22.6274]], device='cuda:0')
#- 4 = stride = 1280/320.

shift_x = (torch.arange(0, 320, device=device) +  0) * 4
shift_y = (torch.arange(0, 320, device=device) +  0) * 4
shift_yy, shift_xx = torch.meshgrid(shift_y, shift_x)
shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
#- shifts  =>
#- tensor([[ 0.,  0.,  0.,  0.],
#-         [ 4.,  0.,  4.,  0.],
#-         [ 8.,  0.,  8.,  0.],
#-         ...
#-         [1272.,  0., 1272.,  0.],
#-         [1276.,  0., 1276.,  0.],
#-         ...
#-         ...
#-         [ 0.,  1276.,  0.,  1276.],
#-         [ 4.,  1276.,  4.,  1276.],
#-         [ 8.,  1276.,  8., 1276.],        
#-         ...
#-         [1272., 1276., 1272., 1276.],
#-         [1276., 1276., 1276., 1276.]], device='cuda:0')        

#- shifts.shape => torch.Size([102400, 4])
all_anchors = base_anchors[None, :, :] + shifts[:, None, :]        
all_anchors = all_anchors.view(-1, 4)
#- all_anchors[:5] =>
#- tensor([[-22.6274, -11.3137,  22.6274,  11.3137],
#-         [-16.0000, -16.0000,  16.0000,  16.0000],
#-         [-11.3137, -22.6274,  11.3137,  22.6274],
#-         [-18.6274, -11.3137,  26.6274,  11.3137],
#-         [-12.0000, -16.0000,  20.0000,  16.0000]], device='cuda:0')        

#- all_anchors.shape (i.e., our anchor boxes) => torch.Size([307200, 4])
</code>
Finally all anchor boxes are concatenated to have shape [409200, 4] i.e., 307200 + 76800 + 19200 + 4800 + 1200 = 409200.

Further, overlap of these anchor boxes with gt bboxes is computed:
<code>
#- https://github.com/open-mmlab/mmdetection/blob/main/mmdet/structures/bbox/bbox_overlaps.py
#- bboxes1.shape, bboxes2.shape => torch.Size([4, 4]), torch.Size([409200, 4])
#- bboxes1 => gt bboxes, bboxes2 => anchor boxes
area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * ( bboxes1[..., 3] - bboxes1[..., 1])
area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * ( bboxes2[..., 3] - bboxes2[..., 1])
lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2]) 
rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])

wh = (rb - lt).clamp(min=0)
overlap = wh[..., 0] * wh[..., 1]
union = area1[..., None] + area2[..., None, :] - overlap

eps = union.new_tensor([eps])
#- eps => tensor([1.0000e-06], device='cuda:0')
union = torch.max(union, eps)
ious = overlap / union
#- ious.shape => torch.Size([4, 409200])
</code>
"ious" is our "overlaps".


From these overlaps, top matching anchor boxes (with gt bboxes) are extracted which have iou score greater than or equal to 0.3. As follows:
<code>
#- https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/assigners/max_iou_assigner.py
max_overlaps, argmax_overlaps = overlaps.max(dim=0)
#- max_overlaps => maximum of "4 iou scores" that anchor boxes have with "4 gt bboxes".
#- argmax_overlaps => index of "gt bbox" with which anchor boxes have maximum score.
#- max_overlaps.shape => torch.Size([409200])        
      
gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)
#- gt_max_overlaps => maximum score from all scores that "gt bboxes" have with "anchor boxes".
#- gt_argmax_overlaps => index of "anchor box" with which "gt bboxes" have maximum score.

#- gt_max_overlaps.shape => torch.Size([4])
#- self.neg_iou_thr => 0.3

assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < self.neg_iou_thr)] = 0
#- assign 0 gt index to scores (from max_overlaps) less than 0.3.

pos_inds = max_overlaps >= self.pos_iou_thr
assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1
#- assign "org gt index"+1 to scores (from max_overlaps) greater than or equal to 0.3.

#- self.match_low_quality => True
if self.match_low_quality:
    for i in range(num_gts):
        #- self.min_pos_iou => 0.3
        #- gt_max_overlaps => maximum score from all scores that "gt bboxes" have with "anchor boxes".
        if gt_max_overlaps[i] >= self.min_pos_iou:
            #- self.gt_max_assign_all => True
            if self.gt_max_assign_all:
                max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                assigned_gt_inds[max_iou_inds] = i + 1
                #- assign "org gt index"+1 to scores (from max_overlaps) which have match with max value at "org gt index" from scores (from overlaps).
</code> 

Further, rpn classification loss (i.e., loss_rpn_cls) is computed using binary_cross_entropy_with_logits between raw "classification scores" and labels, at each scale [1/4, 1/8, 1/16, 1/32, 1/64]. For example for scale 1/4, labels ([307200]) have value 1 where there is match between anchor boxes and gt bboxes elsewhere 0. Raw "classification scores" are reshaped to [BS, 307200] from [BS, 3, 320, 320], before calculation.

<b>[BS, 12, 320, 320], [BS, 12, 160, 160], ..., and [BS, 12, 20, 20]</b> are our "bbox predictions" at different scales [1/4, 1/8, 1/16, 1/32, 1/64]. Before calculation of "rpn smooth l1 loss", bbox predictions are reshaped to <b>[BS, 307200, 4], [BS, 76800, 4], ..., and [BS, 1200, 4]</b>.

RPN smooth L1 loss (i.e., loss_rpn_bbox) is computed between "bbox predictions" and "anchor boxes", at each scale. As follows:
<code>
#- https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/anchor_head.py
#- https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/losses/smooth_l1_loss.py
#- pred.shape => torch.Size([307200, 4])
#- target.shape => torch.Size([307200, 4]) #- anchor boxes
loss = torch.abs(pred - target)
</code>

Under "mathematical operations" block, RPN predicts 1000 bboxes after NMS with IoU threshold=0.70. NMS is performed as per "def batched_nms()" in file "https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/nms.py". NMS will not be applied between elements of different idxs/scales [1/4, 1/8, 1/16, 1/32, 1/64]. The 5th dimension of bbox have the corresponding "classification score".

During RPN prediction, first top 2000 "bbox predictions" are selected at each level based on predicted "classificaton scores". Second, based on these "bbox predictions" corresponding "anchor boxes" are selected. Actually, "anchor boxes" are our proposed bounding boxes and the "bbox predictions" are network outputs used to shift/scale those boxes. Finally, these "anchor bboxes" are shifted according to "bbox predictions":
<code>
#- https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/coders/delta_xywh_bbox_coder.py
#- deltas.shape => torch.Size([9200, 4])
#- means => [0.0, 0.0, 0.0, 0.0]
means = deltas.new_tensor(means).view(1, -1)
#- means => tensor([[0., 0., 0., 0.]], device='cuda:0', dtype=torch.float16)
stds = deltas.new_tensor(stds).view(1, -1)
#- stds => tensor([[1., 1., 1., 1.]], device='cuda:0', dtype=torch.float16)
denorm_deltas = deltas * stds + means

#- calculating the shift/delta/change required in the center (dxy) and size (dwh).
dxy = denorm_deltas[:, :2]
dwh = denorm_deltas[:, 2:]

#- rois.shape => torch.Size([9200, 4])    
#- num_classes => 1
#- rois.repeat(1, num_classes).shape => torch.Size([9200, 4])

#- "rois" are our anchor boxes.
rois_ = rois.repeat(1, num_classes).reshape(-1, 4)
#- rois_.shape => torch.Size([9200, 4])    
pxy = ((rois_[:, :2] + rois_[:, 2:]) * 0.5) #- center point of each bounding box. 
pwh = (rois_[:, 2:] - rois_[:, :2]) #- width and height of each bounding box. 

#- scaling the center adjustments by the width and height of each bounding box.
dxy_wh = pwh * dxy


max_ratio = np.abs(np.log(wh_ratio_clip))
#- wh_ratio_clip, max_ratio => 0.016, 4.135166556742356
dwh = dwh.clamp(min=-max_ratio, max=max_ratio)

gxy = pxy + dxy_wh
#- .exp() => scale the width and height adjustments, effectively converting them from log-space to linear space.
gwh = pwh * dwh.exp()

#- top-left => subtracting half of the adjusted width and height (gwh * 0.5) from the adjusted center (x, y) coordinates. 
x1y1 = gxy - (gwh * 0.5)

#- bottom-right =>  subtracting half of the adjusted width and height (gwh * 0.5) from the adjusted center (x, y) coordinates. 
x2y2 = gxy + (gwh * 0.5)
bboxes = torch.cat([x1y1, x2y2], dim=-1)
</code>

Furhter, after NMS supplied 1000 "anchor boxes" then in the same fashion as described above overlap of these 1000 with "gt bboxes" is computed. And from these overlaps, top matching anchor boxes (with gt bboxes) are extracted which have iou score greater than or equal to 0.3. 

Prior to feeding "anchor boxes" to "SingleRoIExtractor" block, the 5 indices of last dimension is changed i.e., now the index 0 corresponds to batch_id and rest 4 are bbox coordinates. SingleRoIExtractor first partition the "proposals"/"anchor boxes" to the corresponding feature level in accordance to the "scale" (scale is square root of the product of proposal width and height). SingleRoIExtractor then aligns the feature vectors with the corresponding "proposals" and extracts the RoI from "feature vector". Each proposal is defined by a rectangle in the "feature map/vector" and typically represents a region of interest. SingleRoIExtractor performs its operation by using the "class RoIAlign" of file "https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/roi_align.py".
<code>
#- https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/roi_align.py
roi_align = RoIAlignFunction.apply
roi_align(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio, self.pool_mode, self.aligned)
</code>

Outputs of "Shared2FCBBoxHead" block are our final raw "classification scores ([1000, 4])" and "bbox predictions ([1000, 12])". The 4 indices in dim=1 of classification scores represents "num_classes+1". The 12 indices in dim=1 of bbox predictions represents "num_classes*num_box_coordinates" i.e., 3*4.

Further, final classification loss (i.e., loss_cls) is computed using "cross_entropy" between final raw "classification scores" and labels. 
<code>
labels.unique() => tensor([0, 1, 2, 3], device='cuda:0')
#- 3 (an additional class) is used to represent the "background" or "no object" class.  
#- By adding this background class, the network can differentiate between regions with objects and regions without objects.
</code>

Also, bbox predictions ([1000, 12]) are adjusted to shape ([1000, 3, 4]). Then, from dim=1 that index is chosen which corresponds to labels of corresponding "proposals". And, thus reshaping bbox predictions to ([1000, 4]). Finally, smooth L1 loss (i.e., loss_bbox) is computed between final "bbox predictions" and "proposals". 

The top matching anchor boxes ( [50,5] i.e., having iou_score greater than or equal to 0.3) are only feeded to the 2nd "SingleRoIExtractor" block. And, the rest of the RoI extraction procedure is the same as described above. And, both time output feature vectors have different shapes ([..., 7, 7] and [..., 14, 14]) because we have mentioned this in the configuration file "configs/m0_debug.py".
<code>
...
roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0)
...
roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0)
...
</code>
The top matching anchor boxes ([50,5]) are also referred to as our positives.

The output feature vectors of "FCNMaskHead" block are our "mask predictions". The required shape ([..., 28, 28]) is mentioned in configuration file "configs/m0_debug.py".
<code>
rcnn=dict(..., mask_size=28)
</code>

Gt masks are reshaped to ([50, 28, 28]) from  ([50, 1280, 1280]), with the help "gt bboxes" ([50, 4]).
<code>
#- https://github.com/open-mmlab/mmdetection/blob/main/mmdet/structures/mask/mask_target.py
#- https://github.com/open-mmlab/mmdetection/blob/main/mmdet/structures/mask/structures.py
</code>
Further, mask predictions ([50, 3, 28, 28]) are reshaped to ([50, 28, 28]) by picking that index from dim=1 which corresponds to corresponding "gt mask" label. Finally, mask loss (i.e., loss_mask) is computed using "binary_cross_entropy_with_logits" between "mask predictions ([50, 28, 28])" and gt masks ([50, 28, 28]). 
