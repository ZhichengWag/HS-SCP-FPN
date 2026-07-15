# HS-FPN + SCP ablation config.
#
# This config inherits the tested HS-FPN setup and only overrides the parts
# needed by SCP, so comparison with cascade_rcnn_r50_aitod.py stays clean.

import os

_base_ = ['./cascade_rcnn_r50_aitod.py']

custom_imports = dict(
    imports=[
        'mmdet.datasets.aitod',
        'mmdet.models',
        'mmdet.models.necks.hs_scp_fpn',
        'mmdet.models.detectors.scp_cascade_rcnn',
        'mmdet.datasets.transforms.load_scp_pseudo_labels',
        'mmdet.engine.hooks.set_epoch_info_hook',
    ],
    allow_failed_imports=False)

data_root = os.getenv('AITOD_DATA_ROOT', '/mnt/e/AI-TOD/')
backend_args = None

# Use the original 8-class SCP pseudo labels with the V1 SCP neck/detector.
pseudo_label_num_classes = int(os.getenv('SCP_PSEUDO_LABEL_NUM_CLASSES', '8'))
pseudo_label_root = os.getenv(
    'SCP_PSEUDO_LABEL_ROOT',
    data_root + 'pseudo_labels/trainval')

model = dict(
    type='SCPCascadeRCNN',
    neck=dict(
        _delete_=True,
        type='HS_SCP_FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        ratio=(0.25, 0.25),
        num_semantic_classes=pseudo_label_num_classes,
        scp_attn_dim=64,
        return_semantic_logits=True),
    data_preprocessor=dict(pad_seg=True, seg_pad_value=255),
    scp_distill_loss=dict(
        num_classes=pseudo_label_num_classes,
        loss_weight_max=0.5,
        ignore_index=255,
        total_epochs=12))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='LoadSCPPseudoLabels',
        pseudo_label_root=pseudo_label_root,
        key='label',
        ignore_index=255,
        fallback_to_ignore=False),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(
    batch_size=1,
    dataset=dict(pipeline=train_pipeline))

optim_wrapper = dict(
    accumulative_counts=1,
    optimizer=dict(lr=0.005))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=12)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True, milestones=[8, 11], gamma=0.1),
]

custom_hooks = [dict(type='SetEpochInfoHook'), dict(type='NumClassCheckHook')]

work_dir = '/mnt/e/mmdet5090/work_dirs/cascade_rcnn_r50_aitod_scp_b1_k8_epoch12'
