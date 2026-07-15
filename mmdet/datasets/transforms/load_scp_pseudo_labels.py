# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Optional

import numpy as np
from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadSCPPseudoLabels(BaseTransform):
    """Load SCP pseudo semantic labels from per-image ``.npz`` files.

    The loaded map is written to ``results['gt_seg_map']`` so existing
    MMDetection Resize/Flip/PackDetInputs transforms handle it naturally.
    """

    def __init__(self,
                 pseudo_label_root: str,
                 key: str = 'label',
                 suffix: str = '.npz',
                 ignore_index: int = 255,
                 fallback_to_ignore: bool = True) -> None:
        self.pseudo_label_root = pseudo_label_root
        self.key = key
        self.suffix = suffix
        self.ignore_index = ignore_index
        self.fallback_to_ignore = fallback_to_ignore

    def transform(self, results: dict) -> dict:
        img_path = results.get('img_path', None) or results.get(
            'filename', None)
        if img_path is None:
            raise KeyError('LoadSCPPseudoLabels needs img_path or filename.')

        basename = os.path.splitext(os.path.basename(img_path))[0]
        pseudo_path = os.path.join(self.pseudo_label_root,
                                   basename + self.suffix)

        if os.path.exists(pseudo_path):
            label = self._load_npz(pseudo_path)
        elif self.fallback_to_ignore:
            h, w = self._get_image_shape(results)
            label = np.full((h, w), self.ignore_index, dtype=np.uint8)
        else:
            raise FileNotFoundError(
                f'SCP pseudo label not found: {pseudo_path}')

        if label.ndim == 3 and label.shape[0] == 1:
            label = label.squeeze(0)
        if label.ndim != 2:
            raise ValueError(
                f'SCP pseudo label must be 2D, but got {label.shape} from '
                f'{pseudo_path}.')

        results['gt_seg_map'] = label.astype(np.uint8, copy=False)
        results['ignore_index'] = self.ignore_index
        return results

    def _load_npz(self, path: str) -> np.ndarray:
        with np.load(path) as data:
            if self.key in data:
                return data[self.key]
            if 'arr_0' in data:
                return data['arr_0']
            if data.files:
                return data[data.files[0]]
        raise KeyError(f'No arrays found in SCP pseudo label file: {path}')

    @staticmethod
    def _get_image_shape(results: dict) -> tuple:
        shape: Optional[tuple] = results.get('img_shape', None)
        if shape is None:
            shape = results.get('ori_shape', None)
        if shape is None and 'img' in results:
            shape = results['img'].shape[:2]
        if shape is None:
            raise KeyError(
                'Cannot infer fallback pseudo-label shape from results.')
        return int(shape[0]), int(shape[1])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'pseudo_label_root={self.pseudo_label_root}, '
                f'key={self.key}, suffix={self.suffix}, '
                f'ignore_index={self.ignore_index}, '
                f'fallback_to_ignore={self.fallback_to_ignore})')
