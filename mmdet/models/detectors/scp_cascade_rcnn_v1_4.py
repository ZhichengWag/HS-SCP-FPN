# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import get_box_tensor
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .scp_cascade_rcnn_v1_1 import SCPCascadeRCNNV1_1


@MODELS.register_module()
class SCPCascadeRCNNV1_4(SCPCascadeRCNNV1_1):
    """Cascade R-CNN wrapper with soft object-mask gate supervision."""

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 scp_distill_loss: Optional[dict] = None,
                 gate_loss: Optional[dict] = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            scp_distill_loss=scp_distill_loss,
            gate_loss=gate_loss,
            init_cfg=init_cfg)

        gate_loss = gate_loss or {}
        self.gate_dice_weight = gate_loss.get('dice_weight', 0.5)
        self.gate_bce_weight = gate_loss.get('bce_weight', 0.5)
        self.gate_context_scale = gate_loss.get('context_scale', 3.0)
        self.gate_context_value = gate_loss.get('context_value', 0.35)
        self.gate_box_value = gate_loss.get('box_value', 0.8)
        self.gate_min_sigma = gate_loss.get('min_sigma', 1.5)
        self.gate_fg_bce_weight = gate_loss.get('fg_bce_weight', 1.0)
        self.gate_context_bce_weight = gate_loss.get(
            'context_bce_weight', 0.5)
        self.gate_bg_bce_weight = gate_loss.get('bg_bce_weight', 0.1)

    def _loss_gate_dice(self, gate_maps: Optional[list],
                        batch_data_samples: SampleList,
                        batch_inputs: Tensor) -> Optional[Tensor]:
        if gate_maps is None or not gate_maps:
            return None

        losses = []
        for gate in gate_maps:
            gate = torch.nan_to_num(
                gate, nan=0.0, posinf=1.0,
                neginf=0.0).clamp(0.0, 1.0)
            target = self._build_gate_target(
                gate, batch_data_samples, batch_inputs)
            reduce_dims = tuple(range(1, gate.dim()))
            intersection = (gate * target).sum(dim=reduce_dims)
            denominator = gate.sum(dim=reduce_dims) + target.sum(
                dim=reduce_dims)
            dice_loss = 1 - ((2 * intersection + self.gate_eps) /
                             (denominator + self.gate_eps)).mean()

            bce_weight = torch.where(
                target > 0.7,
                target.new_tensor(float(self.gate_fg_bce_weight)),
                torch.where(
                    target > 0.2,
                    target.new_tensor(float(self.gate_context_bce_weight)),
                    target.new_tensor(float(self.gate_bg_bce_weight))))
            bce_loss = F.binary_cross_entropy(
                gate.clamp(self.gate_eps, 1.0 - self.gate_eps),
                target,
                weight=bce_weight,
                reduction='sum') / bce_weight.sum().clamp_min(self.gate_eps)
            losses.append(self.gate_dice_weight * dice_loss +
                          self.gate_bce_weight * bce_loss)

        return self.gate_loss_weight * sum(losses) / len(losses)

    def _build_gate_target(self, gate: Tensor, batch_data_samples: SampleList,
                           batch_inputs: Tensor) -> Tensor:
        batch_size, _, feat_h, feat_w = gate.shape
        target = gate.new_zeros((batch_size, 1, feat_h, feat_w))
        fallback_h, fallback_w = batch_inputs.shape[-2:]

        grid_y, grid_x = torch.meshgrid(
            torch.arange(feat_h, device=gate.device, dtype=gate.dtype),
            torch.arange(feat_w, device=gate.device, dtype=gate.dtype),
            indexing='ij')

        for img_idx, data_sample in enumerate(batch_data_samples):
            if img_idx >= batch_size or 'gt_instances' not in data_sample:
                continue
            gt_instances = data_sample.gt_instances
            if 'bboxes' not in gt_instances:
                continue
            bboxes = get_box_tensor(gt_instances.bboxes).to(
                device=gate.device, dtype=gate.dtype)
            if bboxes.numel() == 0:
                continue

            input_shape = data_sample.metainfo.get(
                'batch_input_shape', (fallback_h, fallback_w))
            input_h, input_w = int(input_shape[0]), int(input_shape[1])
            scale_x = feat_w / max(float(input_w), 1.0)
            scale_y = feat_h / max(float(input_h), 1.0)

            for bbox in bboxes:
                x1 = bbox[0] * scale_x
                y1 = bbox[1] * scale_y
                x2 = bbox[2] * scale_x
                y2 = bbox[3] * scale_y
                box_w = torch.clamp(x2 - x1, min=1.0)
                box_h = torch.clamp(y2 - y1, min=1.0)
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5

                sigma_x = torch.clamp(
                    box_w * 0.5, min=float(self.gate_min_sigma))
                sigma_y = torch.clamp(
                    box_h * 0.5, min=float(self.gate_min_sigma))
                core = torch.exp(-0.5 * (
                    ((grid_x - cx) / sigma_x)**2 +
                    ((grid_y - cy) / sigma_y)**2))

                ctx_sigma_x = sigma_x * float(self.gate_context_scale)
                ctx_sigma_y = sigma_y * float(self.gate_context_scale)
                context = float(self.gate_context_value) * torch.exp(-0.5 * (
                    ((grid_x - cx) / ctx_sigma_x)**2 +
                    ((grid_y - cy) / ctx_sigma_y)**2))

                obj_mask = torch.maximum(core, context)
                ix1 = int(torch.floor(x1).clamp(0, feat_w).item())
                iy1 = int(torch.floor(y1).clamp(0, feat_h).item())
                ix2 = int(torch.ceil(x2).clamp(0, feat_w).item())
                iy2 = int(torch.ceil(y2).clamp(0, feat_h).item())
                if ix2 > ix1 and iy2 > iy1:
                    obj_mask[iy1:iy2, ix1:ix2] = torch.maximum(
                        obj_mask[iy1:iy2, ix1:ix2],
                        obj_mask.new_tensor(float(self.gate_box_value)))
                target[img_idx, 0] = torch.maximum(target[img_idx, 0],
                                                   obj_mask)

        return target.clamp(0.0, 1.0)
