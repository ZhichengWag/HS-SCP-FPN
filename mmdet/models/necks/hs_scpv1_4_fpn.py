# ------------------------------------------------------------------------- #
# HS-SCPV1-4-FPN: SCPV1-3 with soft object-mask guided foreground gates.
#
# V1.4 keeps V1.3's deformable semantic attention and only softens the
# foreground gate activation.  The GT-box-derived soft mask supervision lives
# in the detector wrapper so this neck remains inference-compatible.
# ------------------------------------------------------------------------- #

import torch

from mmdet.registry import MODELS
from .hs_scpv1_1_fpn import ForegroundQueryGate, _finite_clamp
from .hs_scpv1_3_fpn import (ForegroundGatedDeformableSemanticAttention,
                             HFP_SCPV1_3, HS_SCPV1_3_FPN)

__all__ = [
    'HS_SCPV1_4_FPN', 'HFP_SCPV1_4',
    'SoftForegroundQueryGate',
    'SoftForegroundGatedDeformableSemanticAttention'
]


class SoftForegroundQueryGate(ForegroundQueryGate):
    """Foreground gate with temperature and a non-zero lower bound."""

    def __init__(self,
                 feat_channels=256,
                 init_bias=-2.0,
                 temperature=2.0,
                 min_gate=0.05):
        super().__init__(feat_channels=feat_channels, init_bias=init_bias)
        self.temperature = max(float(temperature), 1e-6)
        self.min_gate = float(min_gate)
        if not 0.0 <= self.min_gate < 1.0:
            raise ValueError('min_gate must be in [0, 1).')

    def forward(self, hfp_feat, semantic_logits):
        with torch.no_grad():
            semantic_logits = _finite_clamp(
                semantic_logits.float(), min_val=-50.0, max_val=50.0)
            semantic_conf = torch.softmax(
                semantic_logits, dim=1).max(dim=1, keepdim=True)[0]
            semantic_conf = semantic_conf.to(dtype=hfp_feat.dtype)
        gate_feat = _finite_clamp(hfp_feat.detach())
        gate_input = torch.cat([gate_feat, semantic_conf], dim=1)
        gate = torch.sigmoid(self.net(gate_input) / self.temperature)
        gate = self.min_gate + (1.0 - self.min_gate) * gate
        return torch.nan_to_num(
            gate, nan=self.min_gate, posinf=1.0,
            neginf=self.min_gate).clamp(self.min_gate, 1.0)


class SoftForegroundGatedDeformableSemanticAttention(
        ForegroundGatedDeformableSemanticAttention):
    """V1.3 deformable semantic attention with a softer query gate."""

    def __init__(self,
                 feat_channels=256,
                 num_classes=8,
                 attn_dim=64,
                 num_points=9,
                 gate_init_bias=-2.0,
                 gate_temperature=2.0,
                 min_gate=0.05):
        super().__init__(
            feat_channels=feat_channels,
            num_classes=num_classes,
            attn_dim=attn_dim,
            num_points=num_points,
            gate_init_bias=gate_init_bias)
        self.query_gate = SoftForegroundQueryGate(
            feat_channels=feat_channels,
            init_bias=gate_init_bias,
            temperature=gate_temperature,
            min_gate=min_gate)


class HFP_SCPV1_4(HFP_SCPV1_3):
    """HFP + SCP branch with soft-gated V1.3 deformable attention."""

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
                 gate_temperature=2.0,
                 min_gate=0.05):
        super().__init__(
            in_channels=in_channels,
            ratio=ratio,
            num_classes=num_classes,
            patch=patch,
            attn_dim=attn_dim,
            num_points=num_points,
            isdct=isdct,
            scp_isdct=scp_isdct,
            gate_init_bias=gate_init_bias)
        self.cross_attn = SoftForegroundGatedDeformableSemanticAttention(
            feat_channels=in_channels,
            num_classes=num_classes,
            attn_dim=attn_dim,
            num_points=num_points,
            gate_init_bias=gate_init_bias,
            gate_temperature=gate_temperature,
            min_gate=min_gate)


@MODELS.register_module()
class HS_SCPV1_4_FPN(HS_SCPV1_3_FPN):
    """HS-SCPV1-3 with soft foreground gates for tiny-object context."""

    def __init__(self,
                 *args,
                 num_semantic_classes=8,
                 scp_attn_dim=64,
                 scp_deform_points=9,
                 scp_use_dct_lowpass=False,
                 gate_init_bias=-2.0,
                 gate_temperature=2.0,
                 min_gate=0.05,
                 **kwargs):
        super().__init__(
            *args,
            num_semantic_classes=num_semantic_classes,
            scp_attn_dim=scp_attn_dim,
            scp_deform_points=scp_deform_points,
            scp_use_dct_lowpass=scp_use_dct_lowpass,
            gate_init_bias=gate_init_bias,
            **kwargs)
        out_channels = self.out_channels
        ratio = kwargs.get('ratio', (0.25, 0.25))

        self.SelfAttn_p4 = HFP_SCPV1_4(
            out_channels,
            ratio=None,
            num_classes=num_semantic_classes,
            patch=(8, 8),
            attn_dim=scp_attn_dim,
            num_points=scp_deform_points,
            isdct=False,
            scp_isdct=False,
            gate_init_bias=gate_init_bias,
            gate_temperature=gate_temperature,
            min_gate=min_gate)
        self.SelfAttn_p3 = HFP_SCPV1_4(
            out_channels,
            ratio=None,
            num_classes=num_semantic_classes,
            patch=(8, 8),
            attn_dim=scp_attn_dim,
            num_points=scp_deform_points,
            isdct=False,
            scp_isdct=False,
            gate_init_bias=gate_init_bias,
            gate_temperature=gate_temperature,
            min_gate=min_gate)
        self.SelfAttn_p2 = HFP_SCPV1_4(
            out_channels,
            ratio=ratio,
            num_classes=num_semantic_classes,
            patch=(8, 8),
            attn_dim=scp_attn_dim,
            num_points=scp_deform_points,
            isdct=True,
            scp_isdct=scp_use_dct_lowpass,
            gate_init_bias=gate_init_bias,
            gate_temperature=gate_temperature,
            min_gate=min_gate)
        self.SelfAttn_p1 = HFP_SCPV1_4(
            out_channels,
            ratio=ratio,
            num_classes=num_semantic_classes,
            patch=(16, 16),
            attn_dim=scp_attn_dim,
            num_points=scp_deform_points,
            isdct=True,
            scp_isdct=scp_use_dct_lowpass,
            gate_init_bias=gate_init_bias,
            gate_temperature=gate_temperature,
            min_gate=min_gate)
