# This file is a modified version of the original:
# https://github.com/tascj/kaggle-hubmap-hacking-the-human-vasculature/blob/main/configs/r0.py

# model settings
norm_cfg = dict(type='BN')

# https://mmdetection.readthedocs.io/en/3.x/user_guides/config.html
# https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#predefined-fields
model = dict(
    type='RTMDetWithMaskHead', # RTMDet => Real-Time Object Detectors.
    data_preprocessor=dict(
        type='DetDataPreprocessor', # ...readthedocs.io/en/latest/api.html#mmdet.models.data_preprocessors.DetDataPreprocessor
        mean=[103.53, 116.28, 123.675], # pixel mean values used to pre-training the pre-trained backbone models, ordered in R, G, B.
        std=[57.375, 57.12, 58.395], # pixel standard deviation values used to pre-training the pre-trained backbone models, ordered in R,G,B.
        bgr_to_rgb=False, # whether to convert image from BGR to RGB # BGR => Blue, Green and Red.
        pad_size_divisor=32, # the size of padded image should be divisible by ``pad_size_divisor``. 
        batch_augments=None), # batch-level augmentations.
    mask_head=dict( # https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.roi_heads.FCNMaskHead
        type='FCNMaskHead',
        num_convs=7, # number of convolutional layers in mask head. 
        in_channels=320, # input channels, should be consistent with the output channels of mask roi extractor.
        conv_out_channels=256, # output channels of the convolutional layer.
        num_classes=1), # number of class to be segmented.
    backbone=dict(
        type='CSPNeXt', # https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.backbones.CSPNeXt
        arch='P5', # architecture of CSPNeXt, from {P5, P6}. Defaults to P5.
        expand_ratio=0.5, # ratio to adjust the number of channels of the hidden layer. Defaults to 0.5.
        deepen_factor=1.33, # depth multiplier, multiply number of blocks in CSP layer by this amount. defaults to 1.0.
        widen_factor=1.25, # width multiplier, multiply number of channels in each layer by this amount. defaults to 1.0
        channel_attention=True, # whether to add channel attention in each stage. defaults to True.
        norm_cfg=norm_cfg, # dictionary to construct and config norm layer. defaults to dict(type=’BN’, requires_grad=True)
        act_cfg=dict(type='SiLU', inplace=True)), # config dict for activation layer. defaults to dict(type=’SiLU’). 
    neck=dict(
        type='CSPNeXtPAFPN', # https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.necks.CSPNeXtPAFPN
        in_channels=[320, 640, 1280], # number of input channels per scale.
        out_channels=320, # number of output channels (used at each scale).
        num_csp_blocks=4, # number of bottlenecks in CSPLayer. Defaults to 3.
        expand_ratio=0.5, # ratio to adjust the number of channels of the hidden layer. default: 0.5.
        norm_cfg=norm_cfg, # config dict for normalization layer. default: dict(type=’BN’)
        act_cfg=dict(type='SiLU', inplace=True)), # config dict for activation layer. default: dict(type=’Swish’)
    bbox_head=dict(
        type='RTMDetSepBNHead', # ...readthedocs.io/en/latest/api.html#mmdet.models.dense_heads.RTMDetSepBNHead
        num_classes=3, # number of categories excluding the background category.
        in_channels=320, # number of channels in the input feature map.
        stacked_convs=2, # number of stacked "ConvModule" Blocks after "neck".
        feat_channels=320,  # feature channels of convolutional layers in the head.
        anchor_generator=dict(
            type='MlvlPointGenerator',  # https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/prior_generators/point_generator.py#L92
            offset=0, # the offset of points, the value is normalized with corresponding stride. defaults to 0.5.
            strides=[8, 16, 32]),  # strides of anchors in multiple feature levels in order (w, h).
        bbox_coder=dict(type='DistancePointBBoxCoder'), # https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/coders/distance_point_bbox_coder.py#L9
        loss_cls=dict(
            type='QualityFocalLoss', # https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.losses.QualityFocalLoss
            use_sigmoid=True, # whether sigmoid operation is conducted in QFL. defaults to True.
            beta=2.0, # the beta parameter for calculating the modulating factor. defaults to 2.0.
            loss_weight=1.0), # loss weight of current loss.
        loss_bbox=dict( # config of loss function for the regression branch.
            type='GIoULoss', # https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.losses.GIoULoss
            loss_weight=2.0), # loss weight of the regression branch.
        with_objectness=False, # whether to add an objectness branch. defaults to True.
        exp_on_reg=True, # whether to use .exp() in regression.
        share_conv=True, # whether to share conv layers between stages. defaults to True.
        pred_kernel_size=1, # kernel size of prediction layer. defaults to 1.
        norm_cfg=norm_cfg, # config dict for normalization layer. defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg=dict(type='SiLU', inplace=True)), # config dict for activation layer. defaults to dict(type='SiLU').
    
    train_cfg=dict( # training config of anchor head.
        mask_pos_mode='weighted_sum', # 
        mask_roi_size=28,
        assigner=dict(
            type='IgnoreMaskDynamicSoftLabelAssigner', # https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/assigners/dynamic_soft_label_assigner.py#L40
            topk=13), # select top-k predictions to calculate dynamic k best matches for each gt(ground truth). defaults to 13.
        allowed_border=-1, # the border allowed after padding for valid anchors. 
        pos_weight=-1, # the weight of positive samples during training.
        debug=False),
    
    test_cfg=dict( # testing config of anchor head.
        hflip_tta=False,
        nms_pre=30000, # the number of boxes before NMS (Non-Maximum Suppression).
        min_bbox_size=0, # the allowed minimal box size.
        score_thr=0.001, # threshold to filter out boxes.
        nms=dict( # config of NMS in the second stage.
            type='nms', 
            iou_threshold=0.65), # NMS threshold.
        max_per_img=300), # max number of detections of each image.
    
    init_cfg=dict(
        type='Pretrained',
        
        # https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet
        checkpoint=('https://download.openmmlab.com/mmdetection/v3.0/rtmdet/'
             'rtmdet_x_8xb32-300e_coco/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth')#'work_dirs/r0/epoch_30.pth'  
    )
)

model = dict(
    type='MultiEMADetector',
    momentums=[0.001, 0.0005, 0.00025],
    detector=model,
)

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/'
img_prefix = 'train' # data/
metainfo = dict(classes=('blood_vessel', 'glomerulus', 'unsure'))
backend_args = None

img_scale = (768, 768)

train_pipeline = [
    dict( # first pipeline to load images from file path. 
        type='LoadImageFromFile', backend_args=backend_args),
    dict( # second pipeline to load annotations for current image.
        type='LoadAnnotations', 
        with_bbox=True, # whether to use bounding box, True for detection.
        with_mask=True),
    dict( # third pipeline that resize the images and their annotations.
        type='Resize', 
        scale=(512, 512), # the largest scale of image.
        keep_ratio=True), # whether to keep the ratio between height and width.
    dict( # augmentation pipeline. 
        type='YOLOXHSVRandomAug'), # apply HSV augmentation to image sequentially.
    
    dict( # type is class name RandomRotateScaleCrop ( for the class registered via register_module() ). rest are arguments to the class initialization.
         type='RandomRotateScaleCrop',
         img_scale=img_scale,
         angle_range=(-180, 180),
         scale_range=(0.1, 2.0),
         border_value=(114, 114, 114),
         rotate_prob=0.5,
         scale_prob=1.0,
         hflip_prob=0.5,
         rot90_prob=1.0,
         mask_dtype='u1',
    ),
    dict(type='CropGtMasks', roi_size=56),
    dict( # pipeline that formats the annotation data and decides which keys in the data should be packed into data_samples.
        type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataset1 = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='dtrain0i.json',
    data_prefix=dict(img=img_prefix),
    metainfo=metainfo,
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)

train_dataset2 = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='dtrain_dataset2_dropdup.json',
    data_prefix=dict(img=img_prefix),
    metainfo=metainfo,
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)

train_dataloader = dict(
    batch_size=2, # 8
    num_workers=8, # 4
    persistent_workers=True, # if ``True``, the dataloader will not shut down the worker processes after an epoch end, which can accelerate training speed.
    
#     sampler=dict(  # training data sampler
#         type='DefaultSampler',  # DefaultSampler which supports both distributed and non-distributed training. Refer to https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.dataset.DefaultSampler.html#mmengine.dataset.DefaultSampler
#         shuffle=False),  # randomly shuffle the training data in each epoch    
    
    sampler=dict(
        # GroupMultiSourceSampler randomly selects a group according to the overall aspect ratio distribution of the images in the labeled dataset and the unlabeled dataset, and then sample data to form batches from the two datasets according to source_ratio, so labeled datasets and unlabeled datasets have different repetitions. GroupMultiSourceSampler also ensures that the images in the same batch have similar aspect ratios.
        type='GroupMultiSourceSampler',
        batch_size=2, # 8
        source_ratio=[3, 5]), # the sampling ratio of different source datasets in a mini-batch. source_ratio controls the proportion of labeled data and unlabeled data in the batch.
    
    dataset=dict(
        type='ConcatDataset',
        datasets=[train_dataset1, train_dataset2]),
    )

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False, # whether to drop the last incomplete batch, if the dataset size is not divisible by the batch size. 
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='dval0i.json',
        data_prefix=dict(img=img_prefix),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='FastCocoMetric',
        ann_file=data_root + val_dataloader['dataset']['ann_file'],
        metric=['bbox', 'segm'], # metrics to be evaluated, `bbox` for detection and `segm` for instance segmentation. 
        classwise=True,
        format_only=False,
        backend_args=backend_args),
]
test_evaluator = val_evaluator

# training schedule for 1x
imgs_per_epoch = 338#338  # dataset 1
iters_per_epoch = imgs_per_epoch // 3
train_cfg = dict(type='IterBasedTrainLoop',
                 max_iters=200 * iters_per_epoch,
                 val_interval=iters_per_epoch * 9)
val_cfg = dict(type='MultiEMAValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper', # optimizer wrapper type, switch to AmpOptimWrapper to enable mixed precision training. 
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.01),
    
    paramwise_cfg=dict( # allows users to set hyperparameters in different parts of the model directly by setting the paramwise_cfg.
        norm_decay_mult=0, # weight decay factor to 0 for the weight and bias of the normalization layer to implement the trick of not decaying the weight of the normalization layer.
        bias_decay_mult=0, # weight decay coefficient of the bias (excluding bias of normalization layer and offset of the deformable convolution)..
        bypass_duplicate=True) # whether to skip duplicate parameters, default to False.
    )

auto_scale_lr = dict(enable=True, base_batch_size=16)

param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.001, # the ratio of the starting learning rate used for warmup. 
        by_epoch=False, # the warmup learning rate is updated by iteration.
        begin=0, # start from the first iteration. 
        end=50), # end the warmup at the 500th iteration. 
]

default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'), # update the time spent during iteration into message hub. 
    logger=dict(type='LoggerHook', interval=50), # collect logs from different components of Runner and write them to terminal, JSON file, tensorboard and wandb .etc.
    param_scheduler=dict(type='ParamSchedulerHook'),  # update some hyper-parameters of optimizer. 
    checkpoint=dict( # save checkpoints periodically. 
        type='CheckpointHook', 
        by_epoch=False,
        interval=train_cfg['val_interval'],
        save_optimizer=False),
    sampler_seed=dict(type='DistSamplerSeedHook'), # ensure distributed Sampler shuffle is active.
    visualization=dict(type='DetVisualizationHook')) # detection Visualization Hook. Used to visualize validation and testing process prediction results.

# custom_hooks is a list of all other hook configs. Users can develop their own hooks and insert them in this field.
custom_hooks = [
    dict(type='MultiEMAHook',
         skip_buffers=False,
         interval=1)
]

env_cfg = dict(
    cudnn_benchmark=False, # whether to enable cudnn benchmark.
    mp_cfg=dict( # multi-processing config. 
        mp_start_method='fork', # Use fork to start multi-processing threads. 'fork' usually faster than 'spawn' but maybe unsafe. See discussion in https://github.com/pytorch/pytorch/issues/1355
        opencv_num_threads=0), #  disable opencv multi-threads to avoid system being overloaded.
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
# visualizer to visualize and store the state and intermediate results of the model training and testing process.
visualizer = dict( # https://mmengine.readthedocs.io/en/latest/advanced_tutorials/visualization.html
    type='DetLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer')

log_processor = dict(
    type='LogProcessor', # log processor to process runtime logs.
    window_size=50, # smooth interval of log values. 
    by_epoch=False) #  whether to format logs with epoch type. should be consistent with the train loop's type.

# logging levels => DEBUG, INFO, WARNING, ERROR, CRITICAL
log_level = 'CRITICAL' # default: 'INFO'

load_from = None  # load model checkpoint as a pre-trained model from a given path. this will not resume training.

# whether to resume from the checkpoint defined in `load_from`. if `load_from` is None, it will resume the latest checkpoint in the `work_dir`.
resume = False

custom_imports = dict(imports=['custom_modules'], allow_failed_imports=False)
