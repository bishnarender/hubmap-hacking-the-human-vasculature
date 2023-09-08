# This file is a modified version of the original:
# https://github.com/tascj/kaggle-hubmap-hacking-the-human-vasculature/blob/main/tools/prepare_data.py

import sys
import json
import os.path as osp
import numpy as np
import pandas as pd
import pycocotools.mask as mask_utils
import mmengine
from sklearn.model_selection import StratifiedKFold

DATA_ROOT = 'data/train'

def load_annotations(ann_file):
    ret = {}
    with open(ann_file) as f:
        for line in f:
            # line => {"id":"0006ff2aa7cd","annotations":[{"type":"glomerulus","coordinates":[[[167,249], ...[167,249]]]},
            #    {"type":"blood_vessel","coordinates":[[[283,109], ..., [283,109]]]}
            #    ...
            #    {"type":"blood_vessel","coordinates":[[[227,299], ..., [227,299]]]}]}
            
            # json.loads(s, ...) => deserialize s (a str, bytes or bytearray instance containing a JSON document) to a Python object. i.e., if the json string represents an "json object (dictionary)", json.loads() will return a Python dictionary. if the json string represents an "json array (list)", the function will return a Python list.
            ann = json.loads(line)
            
            # ann => {'id': '0006ff2aa7cd', 'annotations': [{'type': 'glomerulus', 'coordinates': [[[167, 249], ...
            ret[ann['id']] = ann['annotations']
            
    return ret


def decode_coords(coords):
    # .frPyObjects( [pyObjects], h, w ) => convert polygon, bbox, and uncompressed RLE to encoded RLE (Run Length Encoding) mask.
    
    # _.shape => (261, 2)
    # _.flatten().shape => (522,) 
    # len( [_.flatten().tolist() for _ in np.asarray(coords)] ) => 0
    # len( [_.flatten().tolist() for _ in np.asarray(coords)][0] )  => 522
    
    # [_.flatten().tolist() for _ in np.asarray(coords)][0] => [169,228, 168,228, 167,228, ....]
    rles = mask_utils.frPyObjects([_.flatten().tolist() for _ in np.asarray(coords)], 512, 512)
    # rles => [{'size': [512, 512], 'counts': b'YeW2`0^?<^O:L5J6L3L3N2..............R1o0K6I9I:B;JkZU5'}]

    rle = mask_utils.merge(rles)
    # rles => {{'size': [512, 512], 'counts': b'YeW2`0^?<^O:L5J6L3L3N2.............R1o0K6I9I:B;JkZU5'}}
    # type(rles) => <class 'dict'> 
    
    bbox = mask_utils.toBbox(rle)
    # bbox => [143. 138.  38.  90.]
    # type(bbox) => <class 'numpy.ndarray'>
    
    rle['counts'] = rle['counts'].decode()
    # rle['counts'] => YeW2`0^?<^O:L5J6L3L3N2M2O3J6L3N2N5L3M4K2O1O1O0010O001O1O001N2N3M2M3]O^BdNf=X1bB`N`=R1o0K6I9I:B;JkZU5
    
    return bbox, rle



def df2coco(df, annotations):
    print(df.shape)

    coco = {
        'info': {},
        'categories': [{
            'id': 0,
            'name': 'blood_vessel',
        },{
            'id': 1,
            'name': 'glomerulus',
        },{
            'id': 2,
            'name': 'unsure'
        }]
    }
    img_infos = []
    ann_infos = []
    img_id = 0
    ann_id = 0
    
    for _, row in df.iterrows():
        # row =>
        # id            0033bbc76b6b
        # source_wsi               1
        # dataset                  1
        # i                    10240
        # j                    43008
        # Name: 4, dtype: object        
        
        _id = row['id']
        
        img_info = dict(
            id=img_id,
            width=512,
            height=512,
            file_name=f'{_id}.tif',
        )
        
        anns = annotations[_id]
        # anns => [{'type': 'blood_vessel', 'coordinates': [[[169, 228], [168, 228], ..., [406, 511]]]}]
        # len(anns) => 4
        
        for ann in anns:
            if ann['type'] == 'blood_vessel':
                cat_id = 0
            elif ann['type'] == 'glomerulus':
                cat_id = 1
            elif ann['type'] == 'unsure':
                cat_id = 2
            else:
                raise ValueError()
                
            coords = ann['coordinates']
            
            xs = np.asarray(coords)
            # xs.shape => (1, 261, 2)
            
            assert xs.shape[0] == 1
            
            
            # .min(a, axis=None) => return the minimum of an array or minimum along an axis.            
            xmin, ymin = xs[0].min(0)
            # xmin, ymin => 143, 138
            # xs[0,:,0].min() => 143
            
            xmax, ymax = xs[0].max(0)
            w, h = xmax - xmin, ymax - ymin
            
            bbox, rle = decode_coords(coords)
            
            # xs.reshape(1, -1).shape => (1, 522)
            polygon = xs.reshape(1, -1).tolist()
            # xs.reshape(1, -1).tolist() => [[169,228, 168,228, 167,228 ...]]
            
            ann_info = dict(
                id=ann_id,
                image_id=img_id,
                category_id=cat_id,
                iscrowd=0,
                segmentation=polygon,
                area=w * h,
                bbox=[xmin, ymin, w, h],
            )
            ann_infos.append(ann_info)
            ann_id += 1
            
        img_infos.append(img_info)
        img_id += 1        
     
    # img_infos => [{'id': 0, 'width': 512, 'height': 512, 'file_name': '0033bbc76b6b.tif'}]
    coco['images'] = img_infos
    # ann_infos => [{'id': 0, 'image_id': 0, 'category_id': 0, 'iscrowd': 0, 'segmentation': [[169,228, 168,228, ...
    #              169,228]], 'area': 3420, 'bbox': [143, 138, 38, 90]}, ...]
    coco['annotations'] = ann_infos
    return coco



annotations = load_annotations('data/polygons.jsonl') 
df = pd.read_csv('data/tile_meta.csv')
# df.head() =>
#              id  source_wsi  dataset      i      j
# 0  0006ff2aa7cd           2        2  16896  16420
# 1  000e79e206b7           6        3  10240  29184
# 2  00168d1b7522           2        2  14848  14884


wsi1ds1 = df.query('(dataset == 1) and (source_wsi == 1)')
# wsi1ds1.head() =>
#                id  source_wsi  dataset      i      j
# 4    0033bbc76b6b           1        1  10240  43008
# 16   00656c6f2690           1        1  10240  46080
# 33   00d75ad65de3           1        1   8192  39424

wsi2ds1 = df.query('(dataset == 1) and (source_wsi == 2)')


# len(wsi2ds1) => 152
trainval = pd.concat([wsi1ds1, wsi2ds1], axis=0)
# len(trainval) => 422

mmengine.dump(df2coco(trainval, annotations), f'data/dtrainval.json')


# DataFrame.quantile(q=0.5,...) => return values at the given quantile over requested axis.
# wsi1ds1['i'].quantile(0.2) => 3993.600000000002

# split by i
val0 = pd.concat([
    wsi1ds1[wsi1ds1['i'] < wsi1ds1['i'].quantile(0.2)],
    wsi2ds1[wsi2ds1['i'] < wsi2ds1['i'].quantile(0.2)],
], axis=0)

train0 = pd.concat([
    wsi1ds1[wsi1ds1['i'] >= wsi1ds1['i'].quantile(0.2)],
    wsi2ds1[wsi2ds1['i'] >= wsi2ds1['i'].quantile(0.2)],
], axis=0)

mmengine.dump(df2coco(train0, annotations), f'data/dtrain0i.json')
mmengine.dump(df2coco(val0, annotations), f'data/dval0i.json')

# wsi1ds1['i'].quantile(0.8) => 9216.0
# split by i
val1 = pd.concat([
    wsi1ds1[wsi1ds1['i'] > wsi1ds1['i'].quantile(0.8)],
    wsi2ds1[wsi2ds1['i'] > wsi2ds1['i'].quantile(0.8)],
], axis=0)

train1 = pd.concat([
    wsi1ds1[wsi1ds1['i'] <= wsi1ds1['i'].quantile(0.8)],
    wsi2ds1[wsi2ds1['i'] <= wsi2ds1['i'].quantile(0.8)],
], axis=0)

mmengine.dump(df2coco(train1, annotations), f'data/dtrain1i.json')
mmengine.dump(df2coco(val1, annotations), f'data/dval1i.json')

df3 = df.query('(dataset == 2)')
mmengine.dump(df2coco(df3, annotations), 'data/dtrain_dataset2.json')
