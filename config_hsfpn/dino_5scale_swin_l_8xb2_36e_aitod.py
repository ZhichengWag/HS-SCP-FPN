"""DINO-5scale Swin-L baseline on AI-TOD.

Reported settings: DINO, five feature levels, 36 epochs, Adam with
weight_decay=1e-4, and the DETR multi-scale/crop augmentation.

The remaining values follow MMDetection 3.3's official
``dino-5scale_swin-l_8xb2-36e_coco.py`` baseline: ImageNet-22K initialized
Swin-L, batch size 2 per GPU (16 total on 8 GPUs), lr=1e-4,
backbone lr=1e-5, 6 encoder/decoder layers, 900 matching queries,
two-stage DINO, and 100 dynamic denoising queries. These are adopted
baseline assumptions rather than settings reported specifically for AI-TOD.
"""

import os

_base_ = ['../configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py']

custom_imports = dict(
    imports=['mmdet.datasets.aitod'], allow_failed_imports=False)

data_root = os.path.join(
    os.getenv('AITOD_DATA_ROOT', '/mnt/e/AI-TOD/'), '')
backend_args = None

aitod_metainfo = dict(
    classes=(
        'airplane', 'bridge', 'storage-tank', 'ship', 'swimming-pool',
        'vehicle', 'person', 'wind-mill'),
    palette=[
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
        (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70)
    ])

# Swin-L produces stride 4/8/16/32 maps. ChannelMapper adds the stride-64
# level, so the five levels are P2-P6. Other DINO architecture settings are
# inherited unchanged from the official five-scale baseline.
model = dict(
    bbox_head=dict(num_classes=8),
    # AI-TOD test contains up to 2667 instances per image. Keep the same
    # cap as the existing AI-TOD baselines instead of COCO's default 300.
    test_cfg=dict(max_per_img=3000))

# Keep the inherited DETR/DINO RandomChoice pipeline (multi-scale resize,
# optional random crop, and a final multi-scale resize). AI-TOD images are
# mostly 800x800, but training inputs are deliberately not fixed to 800x800.
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        type='AITODDataset',
        data_root=data_root,
        ann_file='annotations/aitod_trainval_v1.json',
        data_prefix=dict(img='trainval/images/'),
        metainfo=aitod_metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=1)))

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='AITODDataset',
        data_root=data_root,
        ann_file='annotations/aitod_test_v1.json',
        data_prefix=dict(img='test/images/'),
        metainfo=aitod_metainfo,
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(data_root, 'annotations/aitod_test_v1.json'),
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# The requested optimizer is Adam (not the AdamW used by the official DINO
# config). The inherited paramwise_cfg applies lr_mult=0.1 to the backbone.
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True, type='Adam', lr=1e-4, weight_decay=1e-4))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=3, save_best='auto'))

randomness = dict(seed=3407, deterministic=False)
work_dir = './work_dirs/dino_5scale_swin_l_8xb2_36e_aitod'
