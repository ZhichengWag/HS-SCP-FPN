# FCOS R50 baseline for VisDrone2019-DET.
#
# Expected prepared layout:
#   <VISDRONE_DATA_ROOT>/
#     VisDrone2019-DET-train/images/*.jpg
#     VisDrone2019-DET-val/images/*.jpg
#     annotations/visdrone2019_det_{train,val}_coco.json

_base_ = ['./fcos_r50_aitod.py']

import os

custom_imports = dict(imports=['mmdet.models'], allow_failed_imports=False)

data_root = os.getenv('VISDRONE_DATA_ROOT', '/mnt/e/数据集/VisDrone2019/')
train_ann_file = os.getenv('VISDRONE_TRAIN_ANN',
                           'annotations/visdrone2019_det_train_coco.json')
val_ann_file = os.getenv('VISDRONE_VAL_ANN',
                         'annotations/visdrone2019_det_val_coco.json')
train_img_prefix = os.getenv('VISDRONE_TRAIN_IMG_PREFIX',
                             'VisDrone2019-DET-train/images')
val_img_prefix = os.getenv('VISDRONE_VAL_IMG_PREFIX',
                           'VisDrone2019-DET-val/images')
backend_args = None

visdrone_metainfo = dict(
    classes=('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
             'tricycle', 'awning-tricycle', 'bus', 'motor'),
    palette=[(220, 20, 60), (0, 128, 255), (119, 11, 32), (0, 0, 142),
             (0, 0, 230), (106, 0, 228), (0, 60, 100), (0, 80, 100),
             (0, 0, 70), (250, 170, 30)])

model = dict(bbox_head=dict(num_classes=10))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
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
        _delete_=True,
        type='CocoDataset',
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
        _delete_=True,
        type='CocoDataset',
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

work_dir = './work_dirs/fcos_r50_visdrone_b2_epoch12'
