# HS-SCPV1-5 dense semantic ablation for VisDrone2019-DET.
#
# The only architectural change from cascade_rcnn_r50_visdrone_scpv1_5.py is
# removal of the fixed 8x8 pooling in every SCP semantic branch.

_base_ = ['./cascade_rcnn_r50_visdrone_scpv1_5.py']

custom_imports = dict(
    imports=[
        'mmdet.datasets.aitod',
        'mmdet.models',
        'mmdet.models.necks.hs_scpv1_5_dense_fpn',
        'mmdet.models.detectors.scp_cascade_rcnn_v1_1',
        'mmdet.datasets.transforms.load_scp_pseudo_labels',
        'mmdet.engine.hooks.set_epoch_info_hook',
    ],
    allow_failed_imports=False)

model = dict(neck=dict(type='HS_SCPV1_5_DENSE_FPN'))

work_dir = (
    '/mnt/e/mmdet5090/work_dirs/'
    'cascade_rcnn_r50_visdrone_scpv1_5_dense_b2_k150_epoch12')
