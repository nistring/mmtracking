# dataset settings
dataset_type = 'SUITDataset'
data_root = '../data/SUIT/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqResize', img_scale=(500, 300), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqBlurAug'),
    dict(type='SeqBrightnessAug', jitter_range=0.2),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqResize', img_scale=(500, 300), keep_ratio=True),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img'],
        meta_keys=('num_left_ref_imgs', 'frame_stride')),
    dict(type='ConcatVideoReferences'),
    dict(type='MultiImagesToTensor', ref_prefix='ref'),
    dict(type='ToList')
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'data/train/',
        ref_img_sampler=dict(
            ref_indices=[-10,-6,-3,-1,1,3,6,10]),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'data/val/',
        ref_img_sampler=dict(
            ref_indices=[-10,-6,-3,-1]),
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'data/val/',
        ref_img_sampler=dict(
            ref_indices=[-10,-6,-3,-1]),
        pipeline=test_pipeline,
        test_mode=True))
