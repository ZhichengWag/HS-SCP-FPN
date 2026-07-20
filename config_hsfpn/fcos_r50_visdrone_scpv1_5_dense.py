# FCOS R50 + dense HS-SCPV1.5-FPN for VisDrone2019-DET.
#
# This ablation replaces the SCP semantic branch with direct prediction from
# each lateral feature map; it does not use the fixed 8x8 spatial pooling.

_base_ = ['./fcos_r50_visdrone_scpv1_5.py']

custom_imports = dict(
    imports=[
        'mmdet.models',
        'mmdet.models.necks.hs_scpv1_5_dense_fpn',
        'mmdet.models.detectors.scp_fcos',
        'mmdet.datasets.transforms.load_scp_pseudo_labels',
        'mmdet.engine.hooks.set_epoch_info_hook',
    ],
    allow_failed_imports=False)

model = dict(neck=dict(type='HS_SCPV1_5_DENSE_FPN'))

work_dir = './work_dirs/fcos_r50_visdrone_scpv1_5_dense_b2_k150_gate0.1_epoch12'
