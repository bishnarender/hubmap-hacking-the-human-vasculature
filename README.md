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


