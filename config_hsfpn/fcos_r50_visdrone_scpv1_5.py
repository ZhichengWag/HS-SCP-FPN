# FCOS R50 + HS-SCPV1.5-FPN for VisDrone2019-DET.
#
# SCP semantic targets are 150-class ADE20K pseudo labels, while FCOS detects
# the 10 VisDrone object classes.

_base_ = ['./fcos_r50_visdrone.py']

import os

custom_imports = dict(
    imports=[
        'mmdet.models',
        'mmdet.models.necks.hs_scpv1_5_fpn',
        'mmdet.models.detectors.scp_fcos',
        'mmdet.datasets.transforms.load_scp_pseudo_labels',
        'mmdet.engine.hooks.set_epoch_info_hook',
    ],
    allow_failed_imports=False)

data_root = os.getenv('VISDRONE_DATA_ROOT',
                      '/home/zhicheng/SCP/data/VisDrone2019_DET/')
backend_args = None

pseudo_label_num_classes = int(
    os.getenv('SCP_PSEUDO_LABEL_NUM_CLASSES', '150'))
pseudo_label_root = os.getenv(
    'SCP_PSEUDO_LABEL_ROOT',
    os.path.join(data_root, 'pseudo_labels_ade20k', 'train'))

model = dict(
    type='SCPFCOS',
    neck=dict(
        _delete_=True,
        type='HS_SCPV1_5_FPN',
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
    gate_loss=dict(loss_weight=0.1, eps=1e-6))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='LoadSCPPseudoLabels',
        pseudo_label_root=pseudo_label_root,
        key='label',
        ignore_index=255,
        fallback_to_ignore=False),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(batch_size=2, dataset=dict(pipeline=train_pipeline))

custom_hooks = [dict(type='SetEpochInfoHook'), dict(type='NumClassCheckHook')]

work_dir = './work_dirs/fcos_r50_visdrone_scpv1_5_b2_k150_gate0.1_epoch12'
