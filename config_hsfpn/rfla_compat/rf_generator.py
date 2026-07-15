from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn.modules.utils import _pair

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import HorizontalBoxes


@TASK_UTILS.register_module()
class RFGenerator:
    """RFLA receptive-field anchor generator for ResNet-50 FPN."""

    def __init__(self,
                 strides: Union[List[int], List[Tuple[int, int]]],
                 ratios: List[float] = [0.5, 1.0, 2.0],
                 fraction: float = 0.5,
                 fpn_layer: str = 'p3',
                 scales: Optional[List[float]] = [1.0],
                 base_sizes: Optional[List[int]] = None,
                 scale_major: bool = True,
                 octave_base_scale: Optional[int] = None,
                 scales_per_octave: Optional[int] = None,
                 centers: Optional[List[Tuple[float, float]]] = None,
                 center_offset: float = 0.,
                 use_box_type: bool = False) -> None:
        if center_offset != 0:
            assert centers is None
        if not (0 <= center_offset <= 1):
            raise ValueError('center_offset should be in range [0, 1].')
        if centers is not None:
            assert len(centers) == len(strides)

        self.strides = [_pair(stride) for stride in strides]
        self.base_sizes = [min(stride) for stride in self.strides
                           ] if base_sizes is None else base_sizes
        assert len(self.base_sizes) == len(self.strides)

        assert ((octave_base_scale is not None
                 and scales_per_octave is not None) ^ (scales is not None))
        if scales is not None:
            self.scales = torch.Tensor(scales)
        else:
            octave_scales = np.array(
                [2**(i / scales_per_octave) for i in range(scales_per_octave)])
            self.scales = torch.Tensor(octave_scales * octave_base_scale)

        self.fraction = fraction
        self.fpn_layer = fpn_layer
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset
        self.use_box_type = use_box_type
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self) -> List[int]:
        return self.num_base_priors

    @property
    def num_base_priors(self) -> List[int]:
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_levels(self) -> int:
        return len(self.strides)

    def gen_base_anchors(self) -> List[Tensor]:
        all_trfs = self.gen_trf()
        if self.fpn_layer == 'p3':
            self.base_sizes = all_trfs[-5:]
        else:
            self.base_sizes = all_trfs[:5]

        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = self.centers[i] if self.centers is not None else None
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=torch.tensor([1.0]),
                    ratios=self.ratios,
                    center=center))
        return multi_level_base_anchors

    @staticmethod
    def gen_trf() -> List[int]:
        jumps = [1]
        for i in range(7):
            jumps.append(jumps[i] * 2)

        r1 = 1 + (7 - 1) * jumps[0]
        r2 = r1 + (3 - 1) * jumps[1]
        trf_p2 = r2 + (3 - 1) * jumps[2] * 3
        r3 = trf_p2 + (3 - 1) * jumps[2]
        trf_p3 = r3 + (3 - 1) * jumps[3] * 3
        r4 = trf_p3 + (3 - 1) * jumps[3]
        trf_p4 = r4 + (3 - 1) * jumps[4] * 5
        r5 = trf_p4 + (3 - 1) * jumps[4]
        trf_p5 = r5 + (3 - 1) * jumps[5] * 2
        trf_p6 = trf_p5 + (3 - 1) * jumps[6]
        trf_p7 = trf_p6 + (3 - 1) * jumps[7]
        return [trf_p2, trf_p3, trf_p4, trf_p5, trf_p6, trf_p7]

    def gen_single_level_base_anchors(self,
                                      base_size: Union[int, float],
                                      scales: Tensor,
                                      ratios: Tensor,
                                      center: Optional[Tuple[float,
                                                             float]] = None
                                      ) -> Tensor:
        w = base_size * self.fraction
        h = base_size * self.fraction
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        return torch.stack([
            x_center - 0.5 * ws, y_center - 0.5 * hs,
            x_center + 0.5 * ws, y_center + 0.5 * hs
        ],
                           dim=-1)

    def _meshgrid(self,
                  x: Tensor,
                  y: Tensor,
                  row_major: bool = True) -> Tuple[Tensor, Tensor]:
        xx = x.repeat(y.shape[0])
        yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
        if row_major:
            return xx, yy
        return yy, xx

    def grid_priors(self,
                    featmap_sizes: List[Tuple],
                    dtype: torch.dtype = torch.float32,
                    device: Union[str, torch.device] = 'cuda') -> List[Tensor]:
        assert self.num_levels == len(featmap_sizes)
        return [
            self.single_level_grid_priors(
                featmap_size, level_idx=i, dtype=dtype, device=device)
            for i, featmap_size in enumerate(featmap_sizes)
        ]

    def single_level_grid_priors(self,
                                 featmap_size: Tuple[int, int],
                                 level_idx: int,
                                 dtype: torch.dtype = torch.float32,
                                 device: Union[str, torch.device] = 'cuda'
                                 ) -> Tensor:
        base_anchors = self.base_anchors[level_idx].to(device).to(dtype)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = torch.arange(0, feat_w, device=device).to(dtype) * stride_w
        shift_y = torch.arange(0, feat_h, device=device).to(dtype) * stride_h
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        if self.use_box_type:
            all_anchors = HorizontalBoxes(all_anchors)
        return all_anchors

    def valid_flags(self,
                    featmap_sizes: List[Tuple[int, int]],
                    pad_shape: Tuple,
                    device: Union[str, torch.device] = 'cuda') -> List[Tensor]:
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i, featmap_size in enumerate(featmap_sizes):
            anchor_stride = self.strides[i]
            feat_h, feat_w = featmap_size
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / anchor_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / anchor_stride[0])), feat_w)
            flags = self.single_level_valid_flags(
                (feat_h, feat_w), (valid_feat_h, valid_feat_w),
                self.num_base_anchors[i], device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self,
                                 featmap_size: Tuple[int, int],
                                 valid_size: Tuple[int, int],
                                 num_base_anchors: int,
                                 device: Union[str, torch.device] = 'cuda'
                                 ) -> Tensor:
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid[:, None].expand(
            valid.size(0), num_base_anchors).contiguous().view(-1)
