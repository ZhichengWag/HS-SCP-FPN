# HS-FPN + SCP config for Faster R-CNN MobileNetV2 on AI-TOD.
#
# This keeps the Faster R-CNN baseline intact and only overrides the parts
# needed by the original 8-class SCP pseudo-label method.

_base_ = ['./faster_rcnn_mobilenetv2_aitod.py']

custom_imports = dict(
    imports=[
        'mmdet.datasets.aitod',
        'mmdet.models',
        'mmdet.models.necks.hs_scp_fpn',
        'mmdet.models.detectors.scp_faster_rcnn',
        'mmdet.datasets.transforms.load_scp_pseudo_labels',
        'mmdet.engine.hooks.set_epoch_info_hook',
    ],
    allow_failed_imports=False)

data_root = '/mnt/e/AI-TOD/'
backend_args = None
pseudo_label_root = data_root + 'pseudo_labels/trainval'

model = dict(
    type='SCPFasterRCNN',
    neck=dict(
        _delete_=True,
        type='HS_SCP_FPN',
        in_channels=[24, 32, 96, 1280],
        out_channels=256,
        num_outs=5,
        ratio=(0.25, 0.25),
        num_semantic_classes=8,
        scp_attn_dim=64,
        return_semantic_logits=True),
    data_preprocessor=dict(pad_seg=True, seg_pad_value=255),
    scp_distill_loss=dict(
        num_classes=8,
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
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/aitod_trainval_v1.json',
        data_prefix=dict(img='trainval/images'),
        pipeline=train_pipeline))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/aitod_test_v1.json',
        data_prefix=dict(img='test/images')))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/aitod_test_v1.json')
test_evaluator = val_evaluator

custom_hooks = [dict(type='SetEpochInfoHook'), dict(type='NumClassCheckHook')]

work_dir = '/mnt/e/mmdet5090/work_dirs/faster_rcnn_mobilenetv2_aitod_scp'
