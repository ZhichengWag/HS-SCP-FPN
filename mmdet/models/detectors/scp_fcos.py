# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .fcos import FCOS
from .scp_cascade_rcnn_v1_1 import SCPCascadeRCNNV1_1


@MODELS.register_module()
class SCPFCOS(FCOS):
    """FCOS wrapper for SCP neck auxiliary outputs.

    SCP necks return detection features, semantic logits and foreground gate
    maps while training.  This wrapper sends only the detection features to
    ``FCOSHead`` and reuses the established SCP distillation and gate losses.
    During prediction the neck returns normal FPN features, so inference stays
    identical to standard FCOS.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 scp_distill_loss: Optional[dict] = None,
                 gate_loss: Optional[dict] = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        scp_distill_loss = scp_distill_loss or {}
        self.scp_num_classes = scp_distill_loss.get('num_classes', 8)
        self.scp_ignore_index = scp_distill_loss.get('ignore_index', 255)
        self.scp_loss_weight_max = scp_distill_loss.get('loss_weight_max', 0.5)
        self.scp_total_epochs = scp_distill_loss.get('total_epochs', 12)
        self.scp_level_weights = scp_distill_loss.get('level_weights', None)

        gate_loss = gate_loss or {}
        self.gate_loss_weight = gate_loss.get('loss_weight', 0.1)
        self.gate_level_weights = gate_loss.get('level_weights', None)
        self.gate_eps = gate_loss.get('eps', 1e-6)
        self.current_epoch = 0

    # Reuse the loss behavior shared by the existing Cascade R-CNN SCP
    # wrapper so the same configuration has the same auxiliary-loss meaning.
    set_epoch = SCPCascadeRCNNV1_1.set_epoch
    extract_feat = SCPCascadeRCNNV1_1.extract_feat
    _extract_feat_with_semantics = (
        SCPCascadeRCNNV1_1._extract_feat_with_semantics)
    _collect_pseudo_labels = SCPCascadeRCNNV1_1._collect_pseudo_labels
    _loss_scp_distill = SCPCascadeRCNNV1_1._loss_scp_distill
    _loss_gate_dice = SCPCascadeRCNNV1_1._loss_gate_dice
    _weighted_level_mean = staticmethod(
        SCPCascadeRCNNV1_1._weighted_level_mean)
    _build_gate_target = SCPCascadeRCNNV1_1._build_gate_target
    _cosine_weight = SCPCascadeRCNNV1_1._cosine_weight

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        x, semantic_logits, gate_maps = self._extract_feat_with_semantics(
            batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)

        pseudo_labels = self._collect_pseudo_labels(batch_data_samples,
                                                    batch_inputs.device)
        loss_scp = self._loss_scp_distill(semantic_logits, pseudo_labels)
        if loss_scp is not None:
            losses['loss_scp_distill'] = loss_scp

        loss_gate = self._loss_gate_dice(gate_maps, batch_data_samples,
                                         batch_inputs)
        if loss_gate is not None:
            losses['loss_scp_gate'] = loss_gate
        return losses
