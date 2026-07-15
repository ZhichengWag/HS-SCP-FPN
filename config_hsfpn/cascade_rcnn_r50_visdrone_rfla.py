# Cascade R-CNN R50-FPN + RFLA for VisDrone2019-DET.

import os
import sys


def _add_repo_root_to_sys_path():
    candidate = os.getcwd()
    while True:
        if os.path.isdir(
                os.path.join(candidate, 'config_hsfpn', 'rfla_compat')):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return
        parent = os.path.dirname(candidate)
        if parent == candidate:
            return
        candidate = parent


_add_repo_root_to_sys_path()
del _add_repo_root_to_sys_path

_base_ = ['./cascade_rcnn_r50_visdrone.py']

custom_imports = dict(
    imports=[
        'mmdet.datasets.aitod',
        'mmdet.models',
        'mmdet.models.necks.hs_fpn',
        'config_hsfpn.rfla_compat',
    ],
    allow_failed_imports=False)

model = dict(
    rpn_head=dict(
        anchor_generator=dict(
            _delete_=True,
            type='RFGenerator',
            fpn_layer='p2',
            fraction=0.5,
            strides=[4, 8, 16, 32, 64])),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                _delete_=True,
                type='HieAssigner',
                ignore_iof_thr=-1,
                gpu_assign_thr=512,
                iou_calculator=dict(type='BboxDistanceMetric'),
                assign_metric='kl',
                topk=[3, 1],
                ratio=0.9))))

train_dataloader = dict(batch_size=2)

work_dir = '/mnt/e/mmdet5090/work_dirs/cascade_rcnn_r50_visdrone_rfla_b2_epoch12'
