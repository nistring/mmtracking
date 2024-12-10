_base_ = [
    '../../_base_/models/faster_rcnn_r50_dc5.py',
    '../../_base_/datasets/imagenet_vid_SUIT_style.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    type='SELSA',
    detector=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid_20210820_162714-939fd657.pth'  # noqa: E501
        ),
        roi_head=dict(
            type='SelsaRoIHead',
            bbox_roi_extractor=dict(
                type='TemporalRoIAlign',
                num_most_similar_points=2,
                num_temporal_attention_blocks=4,
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=512,
                featmap_strides=[16]),
            bbox_head=dict(
                type='SelsaBBoxHead',
                num_shared_fcs=3,
                num_classes=10,
                aggregator=dict(
                    type='SelsaAggregator',
                    in_channels=1024,
                    num_attention_blocks=16)))))

# optimizer
# optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(
#     _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# optimizer
optimizer = dict(type='AdamW', _delete_=True, lr=1.0e-5, weight_decay=0.001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    step=[2,3,4,5,6],
    gamma=0.5)
# runtime settings
total_epochs = 6
evaluation = dict(metric=['bbox'], interval=1)
