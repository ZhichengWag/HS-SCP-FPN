# SCPV2 phase-B warm-start config.
#
# Load the previous SCP checkpoint, freeze the ResNet backbone, and train the
# new SCPV2 neck / RPN / ROI heads / semantic branch first.

_base_ = ['./cascade_rcnn_r50_aitod_scpv2.py']

# Previous trained checkpoint.  Use load_from instead of resume so optimizer
# state and epoch counters are not restored.
load_from = (
    '/mnt/e/mmdet5090/work_dirs/cascade_rcnn_r50_aitod_scpv2_b_0.005/'
    'best_coco_bbox_mAP_epoch_12.pth')
resume = False

model = dict(
    backbone=dict(
        frozen_stages=4,
        norm_eval=True,
        norm_cfg=dict(type='BN', requires_grad=False)))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.0, decay_mult=0.0))))

work_dir = (
    '/mnt/e/mmdet5090/work_dirs/'
    'cascade_rcnn_r50_aitod_scpv2_freeze_backbone_fromv2')
