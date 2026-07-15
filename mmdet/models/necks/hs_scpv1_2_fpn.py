# ------------------------------------------------------------------------- #
# HS-SCPV1-2-FPN: SCPV1-1 plus foreground-gated DCN residual refinement.
#
# This module is intentionally separate from hs_scpv1_1_fpn.py.  It keeps the
# same detector/loss interface and only replaces the HFP+SCP fusion block.
# ------------------------------------------------------------------------- #

import math

import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import ConvModule
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d

from mmdet.registry import MODELS
from .hs_scp_fpn import HFP, SCP, _make_group_norm
from .hs_scpv1_1_fpn import ForegroundQueryGate, HS_SCPV1_1_FPN

__all__ = [
    'HS_SCPV1_2_FPN', 'HFP_SCPV1_2',
    'ForegroundGatedDCNRefinement',
    'ForegroundGatedSemanticCrossAttentionV1_2'
]


class ForegroundGatedDCNRefinement(nn.Module):
    """Refine the gated semantic residual with foreground-gated DCNv2."""

    def __init__(self, feat_channels=256):
        super().__init__()
        hidden_channels = max(16, feat_channels // 4)
        self.offset_mask = nn.Sequential(
            ConvModule(
                feat_channels * 2 + 1,
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
            nn.Conv2d(hidden_channels, 27, 1))
        self.dcn = ModulatedDeformConv2d(
            feat_channels, feat_channels, 3, padding=1, bias=False)
        self.norm = _make_group_norm(feat_channels, 32)
        self.eta = nn.Parameter(torch.zeros(1))
        nn.init.constant_(self.offset_mask[-1].weight, 0.0)
        nn.init.constant_(self.offset_mask[-1].bias, 0.0)

    def forward(self, hfp_feat, delta_eff, gate):
        offset_mask = self.offset_mask(
            torch.cat([hfp_feat, delta_eff, gate], dim=1))
        offset = offset_mask[:, :18] * gate.repeat(1, 18, 1, 1)
        mask = torch.sigmoid(offset_mask[:, 18:])
        refine = self.dcn(delta_eff.contiguous(), offset, mask)
        refine = self.norm(refine)
        return self.eta.to(dtype=refine.dtype) * refine


class ForegroundGatedSemanticCrossAttentionV1_2(nn.Module):
    """V1.1 gated cross-attention followed by gated DCN refinement."""

    def __init__(self,
                 feat_channels=256,
                 num_classes=8,
                 attn_dim=64,
                 gate_init_bias=-2.0):
        super().__init__()
        self.attn_dim = attn_dim
        self.conv_q = nn.Sequential(
            ConvModule(feat_channels, attn_dim, 1, padding=0, bias=False),
            _make_group_norm(attn_dim, 32))
        self.conv_k = nn.Sequential(
            ConvModule(num_classes, attn_dim, 1, padding=0, bias=False),
            _make_group_norm(attn_dim, 32))
        self.conv_v = nn.Sequential(
            ConvModule(num_classes, attn_dim, 1, padding=0, bias=False),
            _make_group_norm(attn_dim, 32))
        self.out_proj = ConvModule(
            attn_dim, feat_channels, 1, padding=0, bias=False)
        self.query_gate = ForegroundQueryGate(
            feat_channels=feat_channels, init_bias=gate_init_bias)
        self.refine = ForegroundGatedDCNRefinement(
            feat_channels=feat_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hfp_feat, scp_feat, patch_size):
        _, _, h, w = hfp_feat.shape
        ph, pw = patch_size

        gate = self.query_gate(hfp_feat, scp_feat)
        q = self.conv_q(hfp_feat) * gate
        k = self.conv_k(scp_feat)
        v = self.conv_v(scp_feat)

        q = rearrange(
            q,
            'b d (h p1) (w p2) -> (b h w) (p1 p2) d',
            p1=ph,
            p2=pw)
        k = rearrange(
            k,
            'b d (h p1) (w p2) -> (b h w) d (p1 p2)',
            p1=ph,
            p2=pw)
        v = rearrange(
            v,
            'b d (h p1) (w p2) -> (b h w) (p1 p2) d',
            p1=ph,
            p2=pw)

        attn = torch.matmul(q, k) / math.sqrt(self.attn_dim)
        attn = self.softmax(attn)
        out = torch.matmul(attn, v)
        out = rearrange(
            out.transpose(1, 2).contiguous(),
            '(b h w) d (p1 p2) -> b d (h p1) (w p2)',
            p1=ph,
            p2=pw,
            h=h // ph,
            w=w // pw)
        delta_eff = gate * self.out_proj(out)
        return hfp_feat + delta_eff + self.refine(
            hfp_feat, delta_eff, gate), gate


class HFP_SCPV1_2(nn.Module):
    """HFP + SCP branch with V1.2 semantic fusion."""

    def __init__(self,
                 in_channels,
                 ratio,
                 num_classes=8,
                 patch=(8, 8),
                 attn_dim=64,
                 isdct=True,
                 scp_isdct=None,
                 gate_init_bias=-2.0):
        super().__init__()
        scp_isdct = isdct if scp_isdct is None else scp_isdct
        self.hfp = HFP(
            in_channels, ratio=ratio, patch=patch, isdct=isdct)
        self.scp = SCP(
            in_channels,
            ratio=ratio,
            num_classes=num_classes,
            isdct=scp_isdct)
        self.cross_attn = ForegroundGatedSemanticCrossAttentionV1_2(
            feat_channels=in_channels,
            num_classes=num_classes,
            attn_dim=attn_dim,
            gate_init_bias=gate_init_bias)

    def forward(self, x, patch_size):
        hfp_out = self.hfp(x)
        semantic_logits = self.scp(x)
        fused, gate = self.cross_attn(hfp_out, semantic_logits, patch_size)
        return fused, semantic_logits, gate


@MODELS.register_module()
class HS_SCPV1_2_FPN(HS_SCPV1_1_FPN):
    """HS-SCP-FPN with V1.2 foreground-gated DCN refinement."""

    def __init__(self,
                 *args,
                 num_semantic_classes=8,
                 scp_attn_dim=64,
                 scp_use_dct_lowpass=False,
                 gate_init_bias=-2.0,
                 **kwargs):
        super().__init__(
            *args,
            num_semantic_classes=num_semantic_classes,
            scp_attn_dim=scp_attn_dim,
            scp_use_dct_lowpass=scp_use_dct_lowpass,
            gate_init_bias=gate_init_bias,
            **kwargs)
        out_channels = self.out_channels
        ratio = kwargs.get('ratio', (0.25, 0.25))

        self.SelfAttn_p4 = HFP_SCPV1_2(
            out_channels,
            ratio=None,
            num_classes=num_semantic_classes,
            patch=(8, 8),
            attn_dim=scp_attn_dim,
            isdct=False,
            scp_isdct=False,
            gate_init_bias=gate_init_bias)
        self.SelfAttn_p3 = HFP_SCPV1_2(
            out_channels,
            ratio=None,
            num_classes=num_semantic_classes,
            patch=(8, 8),
            attn_dim=scp_attn_dim,
            isdct=False,
            scp_isdct=False,
            gate_init_bias=gate_init_bias)
        self.SelfAttn_p2 = HFP_SCPV1_2(
            out_channels,
            ratio=ratio,
            num_classes=num_semantic_classes,
            patch=(8, 8),
            attn_dim=scp_attn_dim,
            isdct=True,
            scp_isdct=scp_use_dct_lowpass,
            gate_init_bias=gate_init_bias)
        self.SelfAttn_p1 = HFP_SCPV1_2(
            out_channels,
            ratio=ratio,
            num_classes=num_semantic_classes,
            patch=(16, 16),
            attn_dim=scp_attn_dim,
            isdct=True,
            scp_isdct=scp_use_dct_lowpass,
            gate_init_bias=gate_init_bias)
