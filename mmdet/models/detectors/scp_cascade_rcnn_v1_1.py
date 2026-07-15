# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import get_box_tensor
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .cascade_rcnn import CascadeRCNN


@MODELS.register_module()
class SCPCascadeRCNNV1_1(CascadeRCNN):
    """Cascade R-CNN wrapper for HS_SCPV1_1_FPN."""

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
            init_cfg=init_cfg)

        scp_distill_loss = scp_distill_loss or {}
        self.scp_num_classes = scp_distill_loss.get('num_classes', 8)
        self.scp_ignore_index = scp_distill_loss.get('ignore_index', 255)
        self.scp_loss_weight_max = scp_distill_loss.get(
            'loss_weight_max', 0.5)
        self.scp_total_epochs = scp_distill_loss.get('total_epochs', 12)
        self.scp_level_weights = scp_distill_loss.get('level_weights', None)

        gate_loss = gate_loss or {}
        self.gate_loss_weight = gate_loss.get('loss_weight', 0.1)
        self.gate_level_weights = gate_loss.get('level_weights', None)
        self.gate_eps = gate_loss.get('eps', 1e-6)
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        x, _, _ = self._extract_feat_with_semantics(batch_inputs)
        return x

    def _extract_feat_with_semantics(
            self, batch_inputs: Tensor
    ) -> Tuple[Tuple[Tensor], Optional[list], Optional[list]]:
        x = self.backbone(batch_inputs)
        semantic_logits = None
        gate_maps = None
        if self.with_neck:
            neck_out = self.neck(x)
            if (isinstance(neck_out, tuple) and len(neck_out) == 3
                    and isinstance(neck_out[1], list)
                    and isinstance(neck_out[2], list)):
                x, semantic_logits, gate_maps = neck_out
            elif (isinstance(neck_out, tuple) and len(neck_out) == 2
                  and isinstance(neck_out[1], list)):
                x, semantic_logits = neck_out
            else:
                x = neck_out
        return x, semantic_logits, gate_maps

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        x, semantic_logits, gate_maps = self._extract_feat_with_semantics(
            batch_inputs)

        losses = dict()
        pseudo_labels = self._collect_pseudo_labels(
            batch_data_samples, batch_inputs.device)
        loss_scp = self._loss_scp_distill(semantic_logits, pseudo_labels)
        if loss_scp is not None:
            losses['loss_scp_distill'] = loss_scp

        loss_gate = self._loss_gate_dice(gate_maps, batch_data_samples,
                                         batch_inputs)
        if loss_gate is not None:
            losses['loss_scp_gate'] = loss_gate

        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = torch.zeros_like(
                    data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)
        return losses

    def _collect_pseudo_labels(self, batch_data_samples: SampleList,
                               device: torch.device) -> Optional[Tensor]:
        if not batch_data_samples or any(
                'gt_sem_seg' not in sample for sample in batch_data_samples):
            return None

        labels = []
        max_h = 0
        max_w = 0
        for data_sample in batch_data_samples:
            label = data_sample.gt_sem_seg.sem_seg
            if label.dim() == 3 and label.size(0) == 1:
                label = label.squeeze(0)
            elif label.dim() != 2:
                raise ValueError(
                    'SCP pseudo label must have shape (H, W) or (1, H, W), '
                    f'but got {tuple(label.shape)}.')
            label = label.to(device=device, dtype=torch.long)
            labels.append(label)
            max_h = max(max_h, label.shape[-2])
            max_w = max(max_w, label.shape[-1])

        padded = []
        for label in labels:
            pad_h = max_h - label.shape[-2]
            pad_w = max_w - label.shape[-1]
            if pad_h > 0 or pad_w > 0:
                label = F.pad(
                    label,
                    pad=(0, pad_w, 0, pad_h),
                    mode='constant',
                    value=self.scp_ignore_index)
            padded.append(label)
        return torch.stack(padded, dim=0)

    def _loss_scp_distill(self, semantic_logits: Optional[list],
                          pseudo_labels: Optional[Tensor]) -> Optional[Tensor]:
        if semantic_logits is None or pseudo_labels is None:
            return None

        lam = self._cosine_weight()
        losses = []
        for sem_logits in semantic_logits:
            sem_logits = torch.nan_to_num(
                sem_logits, nan=0.0, posinf=50.0,
                neginf=-50.0).clamp(-50.0, 50.0)
            target = F.interpolate(
                pseudo_labels.unsqueeze(1).float(),
                size=sem_logits.shape[-2:],
                mode='nearest').squeeze(1).long()
            invalid = (target < 0) | (
                (target >= self.scp_num_classes)
                & (target != self.scp_ignore_index))
            if invalid.any():
                target = target.clone()
                target[invalid] = self.scp_ignore_index

            valid = target != self.scp_ignore_index
            if valid.any():
                losses.append(
                    F.cross_entropy(
                        sem_logits,
                        target,
                        ignore_index=self.scp_ignore_index,
                        reduction='mean'))
            else:
                losses.append(sem_logits.sum() * 0.0)

        if not losses:
            return None
        return lam * self._weighted_level_mean(losses, self.scp_level_weights)

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
            dice = (2 * intersection + self.gate_eps) / (
                denominator + self.gate_eps)
            losses.append(1 - dice.mean())

        return self.gate_loss_weight * self._weighted_level_mean(
            losses, self.gate_level_weights)

    @staticmethod
    def _weighted_level_mean(losses: list,
                             level_weights: Optional[list]) -> Tensor:
        if level_weights is None:
            return sum(losses) / len(losses)

        values = [float(weight) for weight in level_weights]
        if not values:
            return sum(losses) / len(losses)
        if len(values) < len(losses):
            values.extend([values[-1]] * (len(losses) - len(values)))
        elif len(values) > len(losses):
            values = values[:len(losses)]

        weights = losses[0].new_tensor(values).clamp_min(0.0)
        weight_sum = weights.sum().clamp_min(1e-6)
        weighted = [
            loss * weights[idx] for idx, loss in enumerate(losses)
        ]
        return sum(weighted) / weight_sum

    def _build_gate_target(self, gate: Tensor, batch_data_samples: SampleList,
                           batch_inputs: Tensor) -> Tensor:
        batch_size, _, feat_h, feat_w = gate.shape
        target = gate.new_zeros((batch_size, 1, feat_h, feat_w))
        fallback_h, fallback_w = batch_inputs.shape[-2:]

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

            x1 = torch.floor(bboxes[:, 0] * scale_x).long()
            y1 = torch.floor(bboxes[:, 1] * scale_y).long()
            x2 = torch.ceil(bboxes[:, 2] * scale_x).long()
            y2 = torch.ceil(bboxes[:, 3] * scale_y).long()
            x2 = torch.maximum(x2, x1 + 1)
            y2 = torch.maximum(y2, y1 + 1)

            x1 = x1.clamp(0, feat_w)
            y1 = y1.clamp(0, feat_h)
            x2 = x2.clamp(0, feat_w)
            y2 = y2.clamp(0, feat_h)

            for bx1, by1, bx2, by2 in zip(x1, y1, x2, y2):
                if bx2 > bx1 and by2 > by1:
                    target[img_idx, 0, by1:by2, bx1:bx2] = 1.0
        return target

    def _cosine_weight(self) -> float:
        total_epochs = max(int(self.scp_total_epochs), 1)
        epoch = min(max(int(getattr(self, 'current_epoch', 0)), 0),
                    total_epochs)
        return self.scp_loss_weight_max * 0.5 * (
            1.0 + math.cos(math.pi * epoch / total_epochs))
