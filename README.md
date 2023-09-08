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
1. head: the component for specific tasks, e.g., bbox prediction and mask prediction.
2. neck: the component between backbones and heads, e.g., FPN, PAFPN.
* backbone: usually an FCN (Fully Convolution Network) to extract feature maps, e.g., ResNet, MobileNet, CSPNeXt.
* roi extractor: the part for extracting RoI features from feature maps, e.g., RoI Align.
* loss: the component in head for calculating losses, e.g., FocalLoss, L1Loss, and GHMLoss.

### RTMDet Model
-----

### Mask R-CNN Model
-----


