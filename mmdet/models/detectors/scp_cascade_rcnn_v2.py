# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .cascade_rcnn import CascadeRCNN


@MODELS.register_module()
class SCPCascadeRCNNV2(CascadeRCNN):
    """Cascade R-CNN wrapper that consumes HS_SCPV2_FPN semantic logits.

    The detector behaves like the normal CascadeRCNN when no pseudo semantic
    labels are present.  If ``gt_sem_seg`` exists in the data samples, it adds
    a scheduled SCP distillation loss.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 scp_distill_loss: Optional[dict] = None,
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
        self.scp_warmup_epochs = scp_distill_loss.get('warmup_epochs', 1)
        self.scp_ramp_epochs = scp_distill_loss.get('ramp_epochs', 3)
        self.scp_level_loss_weights = scp_distill_loss.get(
            'level_loss_weights', None)
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        x, _ = self._extract_feat_with_semantics(batch_inputs)
        return x

    def _extract_feat_with_semantics(
            self, batch_inputs: Tensor) -> Tuple[Tuple[Tensor], Optional[list]]:
        x = self.backbone(batch_inputs)
        semantic_logits = None
        if self.with_neck:
            neck_out = self.neck(x)
            if (isinstance(neck_out, tuple) and len(neck_out) == 2
                    and isinstance(neck_out[1], list)):
                x, semantic_logits = neck_out
            else:
                x = neck_out
        return x, semantic_logits

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        x, semantic_logits = self._extract_feat_with_semantics(batch_inputs)

        losses = dict()
        pseudo_labels = self._collect_pseudo_labels(
            batch_data_samples, batch_inputs.device)
        loss_scp = self._loss_scp_distill(semantic_logits, pseudo_labels)
        if loss_scp is not None:
            losses['loss_scp_distill'] = loss_scp

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

        lam = self._scheduled_weight()
        losses = []
        level_weights = self._level_weights(len(semantic_logits),
                                            sem_logits_device=semantic_logits[0].device)
        valid_weight_sum = semantic_logits[0].new_tensor(0.0)
        for level_idx, sem_logits in enumerate(semantic_logits):
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
                level_loss = F.cross_entropy(
                    sem_logits,
                    target,
                    ignore_index=self.scp_ignore_index,
                    reduction='mean')
                losses.append(level_loss * level_weights[level_idx])
                valid_weight_sum = valid_weight_sum + level_weights[level_idx]
            else:
                losses.append(sem_logits.sum() * 0.0)

        if not losses:
            return None
        normalizer = torch.clamp(valid_weight_sum, min=1.0)
        return sum(losses) * (lam / normalizer)

    def _scheduled_weight(self) -> float:
        """Warm up and then hold SCP loss weight.

        The V1 cosine schedule starts from the maximum weight.  SCPV2 uses a
        safer ramp because pseudo semantic labels can be noisy in early
        detection training.
        """
        epoch = max(int(getattr(self, 'current_epoch', 0)), 0)
        warmup_epochs = max(int(self.scp_warmup_epochs), 0)
        ramp_epochs = max(int(self.scp_ramp_epochs), 1)
        if epoch < warmup_epochs:
            return 0.0

        ramp_progress = min(
            (epoch - warmup_epochs + 1) / float(ramp_epochs), 1.0)
        return self.scp_loss_weight_max * ramp_progress

    def _level_weights(self, num_levels: int,
                       sem_logits_device: torch.device) -> Tensor:
        if self.scp_level_loss_weights is None:
            weights = [1.0] * num_levels
        else:
            weights = list(self.scp_level_loss_weights)
            if len(weights) < num_levels:
                weights = weights + [weights[-1]] * (num_levels - len(weights))
            weights = weights[:num_levels]
        return torch.tensor(
            weights, device=sem_logits_device, dtype=torch.float32)
