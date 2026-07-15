# ------------------------------------------------------------------------- #
# HS-SCPV1-3-FPN: SCPV1-1 with foreground-gated deformable semantic attention.
#
# This module is intentionally separate from hs_scpv1_1_fpn.py.  It changes
# the semantic cross-attention itself: each HFP query samples a small learned
# set of SCP key/value locations instead of attending to every token in a
# fixed patch/window.
# ------------------------------------------------------------------------- #

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet.registry import MODELS
from .hs_scp_fpn import HFP, SCP, _make_group_norm
from .hs_scpv1_1_fpn import ForegroundQueryGate, HS_SCPV1_1_FPN

__all__ = [
    'HS_SCPV1_3_FPN', 'HFP_SCPV1_3',
    'ForegroundGatedDeformableSemanticAttention'
]


def _make_reference_offsets(num_points):
    """Build local reference offsets in (dx, dy) pixel units."""
    side = int(math.sqrt(num_points))
    if side * side != num_points:
        raise ValueError('num_points must be a square number, e.g. 4, 9, 16.')
    radius = (side - 1) / 2.0
    offsets = []
    for y in torch.linspace(-radius, radius, side):
        for x in torch.linspace(-radius, radius, side):
            offsets.append((float(x), float(y)))
    return torch.tensor(offsets, dtype=torch.float32)


class ForegroundGatedDeformableSemanticAttention(nn.Module):
    """Single-scale deformable cross-attention from HFP queries to SCP K/V."""

    def __init__(self,
                 feat_channels=256,
                 num_classes=8,
                 attn_dim=64,
                 num_points=9,
                 gate_init_bias=-2.0):
        super().__init__()
        self.attn_dim = attn_dim
        self.num_points = num_points
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

        hidden_channels = max(16, feat_channels // 4)
        self.offset_head = nn.Sequential(
            ConvModule(
                feat_channels + 2,
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
            nn.Conv2d(hidden_channels, 2 * num_points, 1))
        nn.init.constant_(self.offset_head[-1].weight, 0.0)
        nn.init.constant_(self.offset_head[-1].bias, 0.0)
        self.register_buffer(
            'reference_offsets',
            _make_reference_offsets(num_points),
            persistent=False)

    def forward(self, hfp_feat, scp_feat, patch_size=None):
        del patch_size
        b, _, h, w = hfp_feat.shape
        gate = self.query_gate(hfp_feat, scp_feat)
        q = self.conv_q(hfp_feat) * gate
        k = self.conv_k(scp_feat)
        v = self.conv_v(scp_feat)

        with torch.no_grad():
            semantic_conf = torch.softmax(
                scp_feat.float(), dim=1).max(dim=1, keepdim=True)[0]
            semantic_conf = semantic_conf.to(dtype=hfp_feat.dtype)
        offset_input = torch.cat([hfp_feat.detach(), gate, semantic_conf], 1)
        offset = self.offset_head(offset_input)
        offset = offset.view(b, self.num_points, 2, h, w)
        offset = offset * gate.unsqueeze(1)

        ref = self.reference_offsets.to(device=hfp_feat.device,
                                        dtype=hfp_feat.dtype)
        ref = ref.view(1, self.num_points, 2, 1, 1)

        y_base, x_base = torch.meshgrid(
            torch.arange(h, device=hfp_feat.device, dtype=hfp_feat.dtype),
            torch.arange(w, device=hfp_feat.device, dtype=hfp_feat.dtype),
            indexing='ij')
        base = torch.stack([x_base, y_base], dim=0).view(1, 1, 2, h, w)
        coords = base + ref + offset
        if w > 1:
            x_norm = coords[:, :, 0] / (w - 1) * 2 - 1
        else:
            x_norm = coords[:, :, 0] * 0
        if h > 1:
            y_norm = coords[:, :, 1] / (h - 1) * 2 - 1
        else:
            y_norm = coords[:, :, 1] * 0
        grid = torch.stack([x_norm, y_norm], dim=-1)
        grid = grid.permute(0, 3, 1, 4, 2).reshape(
            b, h, self.num_points * w, 2)

        sampled_k = F.grid_sample(
            k.float(),
            grid.float(),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True).to(dtype=k.dtype)
        sampled_v = F.grid_sample(
            v.float(),
            grid.float(),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True).to(dtype=v.dtype)
        sampled_k = sampled_k.view(
            b, self.attn_dim, h, self.num_points, w).permute(
                0, 2, 4, 3, 1)
        sampled_v = sampled_v.view(
            b, self.attn_dim, h, self.num_points, w).permute(
                0, 2, 4, 3, 1)
        q = q.permute(0, 2, 3, 1).unsqueeze(3)

        attn = (q * sampled_k).sum(dim=-1) / math.sqrt(self.attn_dim)
        attn = torch.softmax(attn, dim=-1)
        out = (attn.unsqueeze(-1) * sampled_v).sum(dim=3)
        out = out.permute(0, 3, 1, 2).contiguous()
        delta = self.out_proj(out)
        return hfp_feat + gate * delta, gate


class HFP_SCPV1_3(nn.Module):
    """HFP + SCP branch with V1.3 deformable semantic attention."""

    def __init__(self,
                 in_channels,
                 ratio,
                 num_classes=8,
                 patch=(8, 8),
                 attn_dim=64,
                 num_points=9,
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
        self.cross_attn = ForegroundGatedDeformableSemanticAttention(
            feat_channels=in_channels,
            num_classes=num_classes,
            attn_dim=attn_dim,
            num_points=num_points,
            gate_init_bias=gate_init_bias)

    def forward(self, x, patch_size):
        hfp_out = self.hfp(x)
        semantic_logits = self.scp(x)
        fused, gate = self.cross_attn(hfp_out, semantic_logits, patch_size)
        return fused, semantic_logits, gate


@MODELS.register_module()
class HS_SCPV1_3_FPN(HS_SCPV1_1_FPN):
    """HS-SCP-FPN with foreground-gated deformable semantic attention."""

    def __init__(self,
                 *args,
                 num_semantic_classes=8,
                 scp_attn_dim=64,
                 scp_deform_points=9,
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

        self.SelfAttn_p4 = HFP_SCPV1_3(
            out_channels,
            ratio=None,
            num_classes=num_semantic_classes,
            patch=(8, 8),
            attn_dim=scp_attn_dim,
            num_points=scp_deform_points,
            isdct=False,
            scp_isdct=False,
            gate_init_bias=gate_init_bias)
        self.SelfAttn_p3 = HFP_SCPV1_3(
            out_channels,
            ratio=None,
            num_classes=num_semantic_classes,
            patch=(8, 8),
            attn_dim=scp_attn_dim,
            num_points=scp_deform_points,
            isdct=False,
            scp_isdct=False,
            gate_init_bias=gate_init_bias)
        self.SelfAttn_p2 = HFP_SCPV1_3(
            out_channels,
            ratio=ratio,
            num_classes=num_semantic_classes,
            patch=(8, 8),
            attn_dim=scp_attn_dim,
            num_points=scp_deform_points,
            isdct=True,
            scp_isdct=scp_use_dct_lowpass,
            gate_init_bias=gate_init_bias)
        self.SelfAttn_p1 = HFP_SCPV1_3(
            out_channels,
            ratio=ratio,
            num_classes=num_semantic_classes,
            patch=(16, 16),
            attn_dim=scp_attn_dim,
            num_points=scp_deform_points,
            isdct=True,
            scp_isdct=scp_use_dct_lowpass,
            gate_init_bias=gate_init_bias)
