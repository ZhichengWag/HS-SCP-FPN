# HS-FPN + SCPV1-5 config for VisDrone2019-DET.
#
# Expected prepared layout:
#   /mnt/e/<dataset-dir>/VisDrone2019/
#     VisDrone2019-DET-train/images/*.jpg
#     VisDrone2019-DET-val/images/*.jpg
#     annotations/visdrone2019_det_train_coco.json
#     annotations/visdrone2019_det_val_coco.json
#     pseudo_labels_ade20k/train/*.npz
#
# The official VisDrone DET zip files contain txt annotations. Convert them to
# COCO JSON before training, or override VISDRONE_TRAIN_ANN/VISDRONE_VAL_ANN.

import os

_base_ = ['./cascade_rcnn_r50_aitod_scpv1_5.py']

data_root = os.getenv('VISDRONE_DATA_ROOT',
                      '/mnt/e/\u6570\u636e\u96c6/VisDrone2019/')
train_ann_file = os.getenv('VISDRONE_TRAIN_ANN',
                           'annotations/visdrone2019_det_train_coco.json')
val_ann_file = os.getenv('VISDRONE_VAL_ANN',
                         'annotations/visdrone2019_det_val_coco.json')

train_img_prefix = os.getenv('VISDRONE_TRAIN_IMG_PREFIX',
                             'VisDrone2019-DET-train/images')
val_img_prefix = os.getenv('VISDRONE_VAL_IMG_PREFIX',
                           'VisDrone2019-DET-val/images')

pseudo_label_num_classes = int(os.getenv('SCP_PSEUDO_LABEL_NUM_CLASSES',
                                         '150'))
pseudo_label_root = os.getenv(
    'SCP_PSEUDO_LABEL_ROOT',
    os.path.join(data_root, 'pseudo_labels_ade20k', 'train'))

dataset_type = 'CocoDataset'
backend_args = None

visdrone_metainfo = dict(
    classes=('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
             'tricycle', 'awning-tricycle', 'bus', 'motor'),
    palette=[(220, 20, 60), (0, 128, 255), (119, 11, 32), (0, 0, 142),
             (0, 0, 230), (106, 0, 228), (0, 60, 100), (0, 80, 100),
             (0, 0, 70), (250, 170, 30)])

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        ]),
    neck=dict(num_semantic_classes=pseudo_label_num_classes),
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
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_img_prefix),
        metainfo=visdrone_metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_img_prefix),
        metainfo=visdrone_metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(data_root, val_ann_file),
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

work_dir = (
    '/mnt/e/mmdet5090/work_dirs/'
    'cascade_rcnn_r50_visdrone_scpv1_5_b2_k150_epoch12')
