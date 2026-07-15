# Cascade R-CNN R50 with HS-FPN for VisDrone2019-DET.

_base_ = ['./cascade_rcnn_r50_visdrone.py']

model = dict(
    neck=dict(
        _delete_=True,
        type='HS_FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        ratio=(0.25, 0.25)))

work_dir = '/mnt/e/mmdet5090/work_dirs/cascade_rcnn_r50_visdrone_hsfpn_b2_epoch12'
