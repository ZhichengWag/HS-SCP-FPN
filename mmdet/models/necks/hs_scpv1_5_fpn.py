# ------------------------------------------------------------------------- #
# HS-SCPV1-5-FPN: SCPV1-3 with relative position-aware deformable semantic
# attention.
#
# V1.5 keeps V1.3's foreground gate and deformable K/V sampling, and only adds
# a relative-position scalar bias to the sampled-point attention logits.
# ------------------------------------------------------------------------- #

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from .hs_scp_fpn import HFP, SCP
from .hs_scpv1_3_fpn import (ForegroundGatedDeformableSemanticAttention,
                             HS_SCPV1_3_FPN)

__all__ = [
    'HS_SCPV1_5_FPN', 'HFP_SCPV1_5',
    'PositionAwareForegroundGatedDeformableSemanticAttention'
]


def _axis_sincos_position_encoding(coord, dim, temperature):
    """Encode one continuous coordinate axis with fixed sin-cos features."""
    if dim <= 0:
        return coord.new_empty(*coord.shape, 0)

    num_bands = dim // 2
    pieces = []
    if num_bands > 0:
        bands = torch.arange(
            num_bands, device=coord.device, dtype=coord.dtype)
        bands = temperature ** (-bands / max(num_bands - 1, 1))
        scaled = coord.unsqueeze(-1) * (2.0 * math.pi) * bands
        pieces.extend([scaled.sin(), scaled.cos()])
    if dim % 2 == 1:
        pieces.append(coord.unsqueeze(-1))
    return torch.cat(pieces, dim=-1)


def _relative_sincos_position_encoding(dx, dy, dim, temperature=10000.0):
    """Build fixed 2D sin-cos encoding for relative dx/dy coordinates."""
    x_dim = dim // 2
    y_dim = dim - x_dim
    return torch.cat([
        _axis_sincos_position_encoding(dx, x_dim, temperature),
        _axis_sincos_position_encoding(dy, y_dim, temperature),
    ],
                     dim=-1)


class PositionAwareForegroundGatedDeformableSemanticAttention(
        ForegroundGatedDeformableSemanticAttention):
    """V1.3 deformable semantic attention with relative position bias."""

    def __init__(self,
                 feat_channels=256,
                 num_classes=8,
                 attn_dim=64,
                 num_points=9,
                 gate_init_bias=-2.0,
                 pos_dim=32,
                 pos_temperature=10000.0,
                 invalid_sample_mask=True):
        super().__init__(
            feat_channels=feat_channels,
            num_classes=num_classes,
            attn_dim=attn_dim,
            num_points=num_points,
            gate_init_bias=gate_init_bias)
        if pos_dim <= 0:
            raise ValueError('pos_dim must be positive.')
        self.pos_dim = int(pos_dim)
        self.pos_temperature = float(pos_temperature)
        self.invalid_sample_mask = bool(invalid_sample_mask)
        hidden_channels = max(16, attn_dim // 2)
        self.pos_bias_head = nn.Sequential(
            nn.Linear(self.pos_dim, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1))
        nn.init.constant_(self.pos_bias_head[-1].weight, 0.0)
        nn.init.constant_(self.pos_bias_head[-1].bias, 0.0)

    def _make_position_bias(self, coords, base, h, w):
        rel = coords - base
        rho = math.sqrt(self.num_points) / 2.0
        dx = rel[:, :, 0].permute(0, 2, 3, 1) / rho
        dy = rel[:, :, 1].permute(0, 2, 3, 1) / rho
        pos = _relative_sincos_position_encoding(
            dx.float(), dy.float(), self.pos_dim, self.pos_temperature)
        pos = pos.to(dtype=self.pos_bias_head[0].weight.dtype)
        bias = self.pos_bias_head(pos).squeeze(-1)
        return bias

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
        pos_bias = self._make_position_bias(coords, base, h, w)

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
        attn = attn + pos_bias.to(dtype=attn.dtype)
        if self.invalid_sample_mask:
            valid = ((coords[:, :, 0] >= 0) & (coords[:, :, 0] <= w - 1) &
                     (coords[:, :, 1] >= 0) & (coords[:, :, 1] <= h - 1))
            valid = valid.permute(0, 2, 3, 1)
            attn = attn.masked_fill(~valid, -50.0)
        attn = torch.softmax(attn, dim=-1)
        out = (attn.unsqueeze(-1) * sampled_v).sum(dim=3)
        out = out.permute(0, 3, 1, 2).contiguous()
        delta = self.out_proj(out)
        return hfp_feat + gate * delta, gate


class HFP_SCPV1_5(nn.Module):
    """HFP + SCP branch with V1.5 position-aware deformable attention."""

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
        super().__init__()
        scp_isdct = isdct if scp_isdct is None else scp_isdct
        self.hfp = HFP(
            in_channels, ratio=ratio, patch=patch, isdct=isdct)
        self.scp = SCP(
            in_channels,
            ratio=ratio,
            num_classes=num_classes,
            isdct=scp_isdct)
        self.cross_attn = PositionAwareForegroundGatedDeformableSemanticAttention(
            feat_channels=in_channels,
            num_classes=num_classes,
            attn_dim=attn_dim,
            num_points=num_points,
            gate_init_bias=gate_init_bias,
            pos_dim=pos_dim,
            pos_temperature=pos_temperature,
            invalid_sample_mask=invalid_sample_mask)

    def forward(self, x, patch_size):
        hfp_out = self.hfp(x)
        semantic_logits = self.scp(x)
        fused, gate = self.cross_attn(hfp_out, semantic_logits, patch_size)
        return fused, semantic_logits, gate


@MODELS.register_module()
class HS_SCPV1_5_FPN(HS_SCPV1_3_FPN):
    """HS-SCPV1-3 plus relative position bias in deformable attention."""

    def __init__(self,
                 *args,
                 num_semantic_classes=8,
                 scp_attn_dim=64,
                 scp_deform_points=9,
                 scp_use_dct_lowpass=False,
                 gate_init_bias=-2.0,
                 scp_pos_dim=32,
                 scp_pos_temperature=10000.0,
                 scp_invalid_sample_mask=True,
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

        self.SelfAttn_p4 = HFP_SCPV1_5(
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
        self.SelfAttn_p3 = HFP_SCPV1_5(
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
        self.SelfAttn_p2 = HFP_SCPV1_5(
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
        self.SelfAttn_p1 = HFP_SCPV1_5(
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
