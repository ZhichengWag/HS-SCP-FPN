# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Optional

import numpy as np
from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadSCPV2PseudoLabels(BaseTransform):
    """Load SCPV2 pseudo semantic labels from per-image ``.npz`` files.

    The loaded map is written to ``results['gt_seg_map']`` so existing
    MMDetection Resize/Flip/PackDetInputs transforms handle it naturally.
    """

    def __init__(self,
                 pseudo_label_root: str,
                 key: str = 'label',
                 confidence_key: str = 'confidence',
                 confidence_thr: Optional[float] = None,
                 suffix: str = '.npz',
                 ignore_index: int = 255,
                 fallback_to_ignore: bool = True,
                 label_dtype: str = 'int32') -> None:
        self.pseudo_label_root = pseudo_label_root
        self.key = key
        self.confidence_key = confidence_key
        self.confidence_thr = confidence_thr
        self.suffix = suffix
        self.ignore_index = ignore_index
        self.fallback_to_ignore = fallback_to_ignore
        self.label_dtype = label_dtype

    def transform(self, results: dict) -> dict:
        img_path = results.get('img_path', None) or results.get(
            'filename', None)
        if img_path is None:
            raise KeyError(
                'LoadSCPV2PseudoLabels needs img_path or filename.')

        basename = os.path.splitext(os.path.basename(img_path))[0]
        pseudo_path = os.path.join(self.pseudo_label_root,
                                   basename + self.suffix)

        if os.path.exists(pseudo_path):
            label, confidence = self._load_npz(pseudo_path)
        elif self.fallback_to_ignore:
            h, w = self._get_image_shape(results)
            label = np.full((h, w), self.ignore_index, dtype=np.int32)
            confidence = None
        else:
            raise FileNotFoundError(
                f'SCPV2 pseudo label not found: {pseudo_path}')

        if label.ndim == 3 and label.shape[0] == 1:
            label = label.squeeze(0)
        if label.ndim != 2:
            raise ValueError(
                f'SCPV2 pseudo label must be 2D, but got {label.shape} from '
                f'{pseudo_path}.')

        if confidence is not None:
            if confidence.ndim == 3 and confidence.shape[0] == 1:
                confidence = confidence.squeeze(0)
            if confidence.shape != label.shape:
                raise ValueError(
                    'SCPV2 confidence map must have the same shape as label, '
                    f'but got {confidence.shape} vs {label.shape} from '
                    f'{pseudo_path}.')
            if self.confidence_thr is not None:
                label = label.copy()
                label[confidence < self.confidence_thr] = self.ignore_index

        if self.label_dtype is not None:
            label = label.astype(self.label_dtype, copy=False)
        elif not np.issubdtype(label.dtype, np.integer):
            label = label.astype(np.int64, copy=False)

        results['gt_seg_map'] = label
        results['ignore_index'] = self.ignore_index
        return results

    def _load_npz(self, path: str) -> tuple:
        with np.load(path) as data:
            if self.key in data:
                label = data[self.key]
            elif 'arr_0' in data:
                label = data['arr_0']
            elif data.files:
                label = data[data.files[0]]
            else:
                raise KeyError(
                    f'No arrays found in SCP pseudo label file: {path}')

            confidence = None
            if self.confidence_key in data:
                confidence = data[self.confidence_key]
            return label, confidence

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
                f'key={self.key}, confidence_key={self.confidence_key}, '
                f'confidence_thr={self.confidence_thr}, '
                f'suffix={self.suffix}, '
                f'ignore_index={self.ignore_index}, '
                f'fallback_to_ignore={self.fallback_to_ignore}, '
                f'label_dtype={self.label_dtype})')
