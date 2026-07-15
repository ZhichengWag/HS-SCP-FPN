# HS-SCPV1-5 + RFLA RPN config.
#
# This keeps the SCPV1-5 detector/neck and adds RFLA's receptive-field anchor
# generator plus hierarchical KLD label assignment for the RPN.

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

_base_ = ['./cascade_rcnn_r50_aitod_scpv1_5.py']

custom_imports = dict(
    imports=[
        'mmdet.datasets.aitod',
        'mmdet.models',
        'mmdet.models.necks.hs_scpv1_5_fpn',
        'mmdet.models.detectors.scp_cascade_rcnn_v1_1',
        'mmdet.datasets.transforms.load_scp_pseudo_labels',
        'mmdet.engine.hooks.set_epoch_info_hook',
        'config_hsfpn.rfla_compat',
    ],
    allow_failed_imports=False)

data_root = os.getenv('AITOD_DATA_ROOT', '/mnt/e/AI-TOD/')
pseudo_label_root = os.getenv(
    'SCP_PSEUDO_LABEL_ROOT',
    data_root + 'pseudo_labels_ade20k/trainval')

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

optim_wrapper = dict(
    accumulative_counts=1,
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=5000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=12)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=4))

randomness = dict(seed=3407, deterministic=False)

work_dir = '/mnt/e/mmdet5090/work_dirs/cascade_rcnn_r50_aitod_scpv1_5_rfla_b2_lr005_epoch12'
