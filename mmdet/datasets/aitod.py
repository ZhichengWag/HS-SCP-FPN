# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class AITODDataset(CocoDataset):
    """AI-TOD dataset in COCO annotation format."""

    METAINFO = {
        'classes':
        ('airplane', 'bridge', 'storage-tank', 'ship', 'swimming-pool',
         'vehicle', 'person', 'wind-mill'),
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
         (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70)]
    }
