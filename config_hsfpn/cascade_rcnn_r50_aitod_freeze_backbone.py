# Baseline phase-B config with a frozen ResNet backbone.
#
# Load the previous baseline checkpoint, freeze the ResNet backbone, and train
# the HS-FPN / RPN / ROI heads with the same schedule as the baseline config.

_base_ = ['./cascade_rcnn_r50_aitod.py']

model = dict(
    backbone=dict(
        frozen_stages=4,
        norm_eval=True,
        norm_cfg=dict(type='BN', requires_grad=False),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.0, decay_mult=0.0))))

# Previous trained baseline checkpoint. Use load_from instead of resume so
# optimizer state and epoch counters are not restored.
load_from = (
    '/mnt/e/mmdet5090/work_dirs/cascade_rcnn_r50_aitod/'
    'best_coco_bbox_mAP_epoch_12.pth')
resume = False

work_dir = '/mnt/e/mmdet5090/work_dirs/cascade_rcnn_r50_aitod_freeze_backbone'
