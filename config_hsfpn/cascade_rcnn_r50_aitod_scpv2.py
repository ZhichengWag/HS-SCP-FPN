# HS-FPN + SCP ablation config.
#
# This config inherits the tested HS-FPN setup and only overrides the parts
# needed by SCP, so comparison with cascade_rcnn_r50_aitod.py stays clean.

_base_ = ['./cascade_rcnn_r50_aitod.py']

import os

custom_imports = dict(
    imports=[
        'mmdet.datasets.aitod',
        'mmdet.models',
        'mmdet.models.necks.hs_scpv2_fpn',
        'mmdet.models.detectors.scp_cascade_rcnn_v2',
        'mmdet.datasets.transforms.load_scpv2_pseudo_labels',
        'mmdet.engine.hooks.set_epoch_info_hook',
    ],
    allow_failed_imports=False)

data_root = os.getenv('AITOD_DATA_ROOT', '/home/zhicheng/SCP/data/AITOD/')
backend_args = None

# SCPV2 uses the teacher's original semantic category space.
# The default SegFormer-ADE20K teacher has K=150 classes.  Keep this separate
# from the V1 coarse-8 pseudo-label directory so V1 experiments are unchanged.
pseudo_label_num_classes = int(
    os.getenv('SCPV2_PSEUDO_LABEL_NUM_CLASSES', '150'))
pseudo_label_root = os.getenv(
    'SCPV2_PSEUDO_LABEL_ROOT',
    data_root + 'pseudo_labels_ade20k/trainval')

model = dict(
    type='SCPCascadeRCNNV2',
    neck=dict(
        _delete_=True,
        type='HS_SCPV2_FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        ratio=(0.25, 0.25),
        num_semantic_classes=pseudo_label_num_classes,
        scp_attn_dim=64,
        scp_pool_sizes=((16, 16), (12, 12), (8, 8), (8, 8)),
        # Phase B: only P1/P2 enable high-frequency detail guidance.
        enable_p12_detail_gate=True,
        detach_detail=True,
        high_to_low_beta_init=0.1,
        fusion_scale_init=0.0,
        return_semantic_logits=True),
    data_preprocessor=dict(pad_seg=True, seg_pad_value=255),
    scp_distill_loss=dict(
        num_classes=pseudo_label_num_classes,
        loss_weight_max=0.5,
        warmup_epochs=1,
        ramp_epochs=3,
        level_loss_weights=[0.5, 0.75, 1.0, 1.0],
        ignore_index=255,
        total_epochs=24))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='LoadSCPV2PseudoLabels',
        pseudo_label_root=pseudo_label_root,
        key='label',
        confidence_key='confidence',
        confidence_thr=None,
        ignore_index=255,
        fallback_to_ignore=False,
        label_dtype='int32'),
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

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=12)

custom_hooks = [dict(type='SetEpochInfoHook'), dict(type='NumClassCheckHook')]

work_dir = '/mnt/e/mmdet5090/work_dirs/cascade_rcnn_r50_aitod_scpv2_k150_0.005_epoch24'
