# HS-FPN + SCPV1-6 cross-level guided gate config.
#
# This config mirrors SCPV1-5, but swaps the neck to HS_SCPV1_6_FPN.  V1.6
# keeps position-aware deformable semantic attention and guides the P1/P2
# foreground gates with the next coarser level's detached gate/confidence.

_base_ = ['./cascade_rcnn_r50_aitod.py']

import os

custom_imports = dict(
    imports=[
        'mmdet.datasets.aitod',
        'mmdet.models',
        'mmdet.models.necks.hs_scpv1_6_fpn',
        'mmdet.models.detectors.scp_cascade_rcnn_v1_1',
        'mmdet.datasets.transforms.load_scp_pseudo_labels',
        'mmdet.engine.hooks.set_epoch_info_hook',
    ],
    allow_failed_imports=False)

data_root = os.getenv('AITOD_DATA_ROOT', '/home/zhicheng/SCP/data/AITOD/')
backend_args = None

pseudo_label_num_classes = int(os.getenv('SCP_PSEUDO_LABEL_NUM_CLASSES', '150'))
pseudo_label_root = os.getenv(
    'SCP_PSEUDO_LABEL_ROOT',
    data_root + 'pseudo_labels_ade20k/trainval')

model = dict(
    type='SCPCascadeRCNNV1_1',
    neck=dict(
        _delete_=True,
        type='HS_SCPV1_6_FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        ratio=(0.25, 0.25),
        num_semantic_classes=pseudo_label_num_classes,
        scp_attn_dim=64,
        scp_deform_points=9,
        scp_pos_dim=32,
        scp_pos_temperature=10000.0,
        scp_invalid_sample_mask=True,
        scp_use_dct_lowpass=False,
        gate_init_bias=-2.0,
        return_semantic_logits=True),
    data_preprocessor=dict(pad_seg=True, seg_pad_value=255),
    scp_distill_loss=dict(
        num_classes=pseudo_label_num_classes,
        loss_weight_max=0.5,
        ignore_index=255,
        total_epochs=12),
    gate_loss=dict(
        loss_weight=0.1,
        eps=1e-6))

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
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1),
]

custom_hooks = [dict(type='SetEpochInfoHook'), dict(type='NumClassCheckHook')]

randomness = dict(seed=3407, deterministic=False)

work_dir = '/mnt/e/mmdet5090/work_dirs/cascade_rcnn_r50_aitod_scpv1_6_b2_k150_t10000_epoch12'
