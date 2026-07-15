# ------------------------------------------------------------------------- #
# HS-SCPV1-7-FPN: V1.5 semantic fusion on all HFP levels.
#
# V1.7 keeps V1.5's position-aware deformable SCP fusion on P1-P4.  The
# difference is that the foreground gate can back-propagate gate loss into the
# HFP branch, while semantic confidence remains detached from the SCP branch.
# ------------------------------------------------------------------------- #

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.registry import MODELS
from .hs_scp_fpn import HFP, SCP, _make_group_norm
from .hs_scpv1_1_fpn import _finite_clamp
from .hs_scpv1_5_fpn import (HS_SCPV1_5_FPN, HFP_SCPV1_5,
                             PositionAwareForegroundGatedDeformableSemanticAttention)

__all__ = [
    'HS_SCPV1_7_FPN', 'HFP_SCPV1_7',
    'HFPGradForegroundQueryGate',
    'HFPGradPositionAwareForegroundGatedDeformableSemanticAttention'
]


class HFPGradForegroundQueryGate(nn.Module):
    """Predict foreground gate without detaching HFP features."""

    def __init__(self, feat_channels=256, init_bias=-2.0):
        super().__init__()
        hidden_channels = max(16, feat_channels // 4)
        self.net = nn.Sequential(
            ConvModule(
                feat_channels + 1,
                hidden_channels,
                1,
                padding=0,
                bias=False),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                3,
                padding=1,
                groups=hidden_channels,
                bias=False),
            _make_group_norm(hidden_channels, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1))
        nn.init.constant_(self.net[-1].bias, float(init_bias))

    def forward(self, hfp_feat, semantic_logits):
        with torch.no_grad():
            semantic_logits = _finite_clamp(
                semantic_logits.float(), min_val=-50.0, max_val=50.0)
            semantic_conf = torch.softmax(
                semantic_logits, dim=1).max(dim=1, keepdim=True)[0]
            semantic_conf = semantic_conf.to(dtype=hfp_feat.dtype)
        gate_feat = _finite_clamp(hfp_feat)
        gate_input = torch.cat([gate_feat, semantic_conf], dim=1)
        gate = torch.sigmoid(self.net(gate_input))
        return torch.nan_to_num(
            gate, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


class HFPGradPositionAwareForegroundGatedDeformableSemanticAttention(
        PositionAwareForegroundGatedDeformableSemanticAttention):
    """V1.5 attention whose gate loss can update HFP features."""

    def __init__(self, *args, gate_init_bias=-2.0, **kwargs):
        super().__init__(*args, gate_init_bias=gate_init_bias, **kwargs)
        self.query_gate = HFPGradForegroundQueryGate(
            feat_channels=kwargs.get('feat_channels', 256),
            init_bias=gate_init_bias)


class HFP_SCPV1_7(HFP_SCPV1_5):
    """HFP + SCP branch with HFP-trainable foreground gates."""

    def __init__(self,
                 in_channels,
                 ratio,
                 num_classes=8,
                 patch=(8, 8),
                 attn_dim=64,
                 num_points=9,
                 isdct=True,
                 scp_isdct=None,
                 gate_init_bias=-2.0,
                 pos_dim=32,
                 pos_temperature=10000.0,
                 invalid_sample_mask=True):
        nn.Module.__init__(self)
        scp_isdct = isdct if scp_isdct is None else scp_isdct
        self.hfp = HFP(
            in_channels, ratio=ratio, patch=patch, isdct=isdct)
        self.scp = SCP(
            in_channels,
            ratio=ratio,
            num_classes=num_classes,
            isdct=scp_isdct)
        self.cross_attn = HFPGradPositionAwareForegroundGatedDeformableSemanticAttention(
            feat_channels=in_channels,
            num_classes=num_classes,
            attn_dim=attn_dim,
            num_points=num_points,
            gate_init_bias=gate_init_bias,
            pos_dim=pos_dim,
            pos_temperature=pos_temperature,
            invalid_sample_mask=invalid_sample_mask)


@MODELS.register_module()
class HS_SCPV1_7_FPN(HS_SCPV1_5_FPN):
    """HS-SCPV1-5 with gate loss allowed to update HFP features."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        out_channels = self.out_channels
        ratio = kwargs.get('ratio', (0.25, 0.25))
        num_semantic_classes = kwargs.get('num_semantic_classes', 8)
        scp_attn_dim = kwargs.get('scp_attn_dim', 64)
        scp_deform_points = kwargs.get('scp_deform_points', 9)
        scp_use_dct_lowpass = kwargs.get('scp_use_dct_lowpass', False)
        gate_init_bias = kwargs.get('gate_init_bias', -2.0)
        scp_pos_dim = kwargs.get('scp_pos_dim', 32)
        scp_pos_temperature = kwargs.get('scp_pos_temperature', 10000.0)
        scp_invalid_sample_mask = kwargs.get('scp_invalid_sample_mask', True)

        self.SelfAttn_p4 = HFP_SCPV1_7(
            out_channels,
            ratio=None,
            num_classes=num_semantic_classes,
            patch=(8, 8),
            attn_dim=scp_attn_dim,
            num_points=scp_deform_points,
            isdct=False,
            scp_isdct=False,
            gate_init_bias=gate_init_bias,
            pos_dim=scp_pos_dim,
            pos_temperature=scp_pos_temperature,
            invalid_sample_mask=scp_invalid_sample_mask)
        self.SelfAttn_p3 = HFP_SCPV1_7(
            out_channels,
            ratio=None,
            num_classes=num_semantic_classes,
            patch=(8, 8),
            attn_dim=scp_attn_dim,
            num_points=scp_deform_points,
            isdct=False,
            scp_isdct=False,
            gate_init_bias=gate_init_bias,
            pos_dim=scp_pos_dim,
            pos_temperature=scp_pos_temperature,
            invalid_sample_mask=scp_invalid_sample_mask)
        self.SelfAttn_p2 = HFP_SCPV1_7(
            out_channels,
            ratio=ratio,
            num_classes=num_semantic_classes,
            patch=(8, 8),
            attn_dim=scp_attn_dim,
            num_points=scp_deform_points,
            isdct=True,
            scp_isdct=scp_use_dct_lowpass,
            gate_init_bias=gate_init_bias,
            pos_dim=scp_pos_dim,
            pos_temperature=scp_pos_temperature,
            invalid_sample_mask=scp_invalid_sample_mask)
        self.SelfAttn_p1 = HFP_SCPV1_7(
            out_channels,
            ratio=ratio,
            num_classes=num_semantic_classes,
            patch=(16, 16),
            attn_dim=scp_attn_dim,
            num_points=scp_deform_points,
            isdct=True,
            scp_isdct=scp_use_dct_lowpass,
            gate_init_bias=gate_init_bias,
            pos_dim=scp_pos_dim,
            pos_temperature=scp_pos_temperature,
            invalid_sample_mask=scp_invalid_sample_mask)
