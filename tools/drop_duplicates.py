# This file is a modified version of the original:
# https://github.com/tascj/kaggle-hubmap-hacking-the-human-vasculature/blob/main/tools/drop_duplicates.py

import mmengine
import pycocotools.mask as mask_utils
from pycocotools.coco import COCO
import sys
import numpy as np

def decode_mask(ann):
    # .frPyObjects( [pyObjects], h, w ) => convert polygon, bbox, and uncompressed RLE to encoded RLE (Run Length Encoding) mask.    
    
    return mask_utils.decode(mask_utils.frPyObjects(ann['segmentation'], 512, 512))

def mask_iou(a, b):
    inter = (a == 1) & (b == 1)
    union = (a == 1) | (b == 1) # True | False => True
    return inter.sum() / union.sum()

coco = COCO('data/dtrain_dataset2.json')
count = 0
valid_ann_ids = []
# coco.getImgIds() => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ..., 1207, 1208, 1209, 1210]

for img_id in coco.getImgIds():
    anns = coco.loadAnns(coco.getAnnIds(img_id))

    # ann => [{'id': 0, 'image_id': 0, 'category_id': 0, 'iscrowd': 0, 'segmentation': [[169,228, 168,228, ...
    #            - 169,228]], 'area': 3420, 'bbox': [143, 138, 38, 90]}, ...]    
    # np.array(ann['segmentation']).shape => (1, 2078)
    # decode_mask(ann).shape => (512, 512, 1)    
    # decode_mask(ann) => [ [[1][1][1] ...[0][0][0]] ... [[0][0][0] ... [0][0][0]]]
    masks = [decode_mask(ann) for ann in anns]
    # len(masks) => 9
    
    ious = {}
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            # masks[i].shape, masks[j].shape => (512, 512, 1), (512, 512, 1)
            iou = mask_iou(masks[i], masks[j])
            ious[(i, j)] = iou
            
    # ious => {(0, 1): 0.0, (0, 2): 0.0, (0, 3): 0.0, (0, 4): 0.0, (0, 5): 0.0, (0, 6): 0.0, (0, 7): 0.0, (0, 8): 0.0, (1, 2): 0.0, (1, 3): 0.0, (1, 4): 0.0, (1, 5): 0.0, ..., (6, 8): 0.0, (7, 8): 0.0}
    # len(ious.items()) => 36
    
    valid_inds = set()
    for candidate in range(len(masks)):
        if len(valid_inds) == 0:
            valid_inds.add(candidate)
        else:
            # valid_inds => {0}           (for candidate=1 if add_new remains True throughout this else)
            # valid_inds => {0, 1}        (for candidate=2 if add_new remains True throughout this else)
            # valid_inds => {0, 1, 2}     (for candidate=3 if add_new remains True throughout this else)
            # 1+2+3+4+5+6+7+8 = 36 matches
            add_new = True
            for valid_ind in valid_inds:
                # valid_ind => 0
                if ious[tuple(sorted([valid_ind, candidate]))] == 1.0:
                    print(f'Image Id "{img_id}", annotation id pair {tuple(sorted([valid_ind, candidate]))} => duplicate')
                    add_new = False
                    break
            if add_new:
                valid_inds.add(candidate)
    

    # valid_inds => {0, 1, 2, 3, 4, 5, 6, 7, 8}
    
    #print(len(masks), len(valid_inds))
    if len(masks) != len(valid_inds):
        print(f'keep annotation ids {valid_inds}.\n')
        count += len(masks) - len(valid_inds)

    for valid_ind in valid_inds:
        valid_ann_ids.append(anns[valid_ind]['id'])
        # valid_ann_ids => [0, 1, 2, 3, 4, 5, ...]



valid_ann_ids = set(valid_ann_ids)

d = mmengine.load('data/dtrain_dataset2.json')
annotations = [
    ann for ann in d['annotations']
    if ann['id'] in valid_ann_ids
]
d['annotations'] = annotations
mmengine.dump(d, 'data/dtrain_dataset2_dropdup.json')
