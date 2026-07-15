# ------------------------------------------------------------------------- #
# HS-SCPV1-1-FPN: HS-SCP-FPN with a foreground-aware query gate.
#
# This file intentionally does not modify hs_scp_fpn.py or hs_scpv2_fpn.py.
# It reuses the original HFP/SCP/SDP blocks and replaces only the semantic
# cross-attention fusion with a gated query/residual path:
#
#   Q_hat = G * Q
#   Y = F + G * Delta(Q_hat, K, V)
# ------------------------------------------------------------------------- #

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn import ConvModule

from mmdet.registry import MODELS
from .hs_scp_fpn import HFP, HS_SCP_FPN, SCP, SDP, _make_group_norm

__all__ = [
    'HS_SCPV1_1_FPN', 'HFP_SCPV1_1',
    'ForegroundQueryGate', 'ForegroundGatedSemanticCrossAttention'
]


def _finite_clamp(x, min_val=-1e4, max_val=1e4):
    return torch.nan_to_num(
        x, nan=0.0, posinf=max_val,
        neginf=min_val).clamp(min_val, max_val)


class ForegroundQueryGate(nn.Module):
    """Predict a spatial foreground gate for SCP query selection."""

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
        gate_feat = _finite_clamp(hfp_feat.detach())
        gate_input = torch.cat([gate_feat, semantic_conf], dim=1)
        gate = torch.sigmoid(self.net(gate_input))
        return torch.nan_to_num(
            gate, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


class ForegroundGatedSemanticCrossAttention(nn.Module):
    """SCP cross-attention with foreground-gated query and residual output."""

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
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hfp_feat, scp_feat, patch_size):
        hfp_feat = _finite_clamp(hfp_feat)
        scp_feat = _finite_clamp(scp_feat, min_val=-50.0, max_val=50.0)
        _, _, h, w = hfp_feat.shape
        ph, pw = patch_size

        gate = self.query_gate(hfp_feat, scp_feat)
        q = _finite_clamp(self.conv_q(hfp_feat)) * gate
        k = _finite_clamp(self.conv_k(scp_feat))
        v = _finite_clamp(self.conv_v(scp_feat))

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

        attn = torch.matmul(q.float(), k.float()) / math.sqrt(self.attn_dim)
        attn = _finite_clamp(attn, min_val=-50.0, max_val=50.0)
        attn = self.softmax(attn)
        out = torch.matmul(attn.to(dtype=v.dtype), v)
        out = rearrange(
            out.transpose(1, 2).contiguous(),
            '(b h w) d (p1 p2) -> b d (h p1) (w p2)',
            p1=ph,
            p2=pw,
            h=h // ph,
            w=w // pw)
        delta = _finite_clamp(self.out_proj(out))
        fused = _finite_clamp(hfp_feat + gate * delta)
        return fused, gate


class HFP_SCPV1_1(nn.Module):
    """HFP + SCP branch with foreground-gated semantic fusion."""

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
        self.cross_attn = ForegroundGatedSemanticCrossAttention(
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
class HS_SCPV1_1_FPN(HS_SCP_FPN):
    """HS-SCP-FPN with foreground-aware query gates.

    In training mode this neck returns ``(outs, semantic_logits, gate_maps)``.
    In evaluation mode it returns only ``outs`` to match normal detector
    inference behavior.
    """

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
            **kwargs)
        out_channels = self.out_channels
        ratio = kwargs.get('ratio', (0.25, 0.25))

        self.SelfAttn_p4 = HFP_SCPV1_1(
            out_channels,
            ratio=None,
            num_classes=num_semantic_classes,
            patch=(8, 8),
            attn_dim=scp_attn_dim,
            isdct=False,
            scp_isdct=False,
            gate_init_bias=gate_init_bias)
        self.SelfAttn_p3 = HFP_SCPV1_1(
            out_channels,
            ratio=None,
            num_classes=num_semantic_classes,
            patch=(8, 8),
            attn_dim=scp_attn_dim,
            isdct=False,
            scp_isdct=False,
            gate_init_bias=gate_init_bias)
        self.SelfAttn_p2 = HFP_SCPV1_1(
            out_channels,
            ratio=ratio,
            num_classes=num_semantic_classes,
            patch=(8, 8),
            attn_dim=scp_attn_dim,
            isdct=True,
            scp_isdct=scp_use_dct_lowpass,
            gate_init_bias=gate_init_bias)
        self.SelfAttn_p1 = HFP_SCPV1_1(
            out_channels,
            ratio=ratio,
            num_classes=num_semantic_classes,
            patch=(16, 16),
            attn_dim=scp_attn_dim,
            isdct=True,
            scp_isdct=scp_use_dct_lowpass,
            gate_init_bias=gate_init_bias)

        self.CrossAtten_p4_p3 = SDP(dim=out_channels)
        self.CrossAtten_p3_p2 = SDP(dim=out_channels)
        self.CrossAtten_p2_p1 = SDP(dim=out_channels)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        used_backbone_levels = len(laterals)
        semantic_logits = []
        gate_maps = []

        if used_backbone_levels == 4:
            _, _, h, w = laterals[3].size()
            patch_size = [h, w]

            laterals[3], sem, gate = self.SelfAttn_p4(
                laterals[3], patch_size)
            semantic_logits.append(sem)
            gate_maps.append(gate)

            p3, sem, gate = self.SelfAttn_p3(laterals[2], patch_size)
            laterals[2] = self.CrossAtten_p4_p3(
                p3, self.fpn_upsample(laterals[3]), patch_size)
            semantic_logits.append(sem)
            gate_maps.append(gate)

            p2, sem, gate = self.SelfAttn_p2(laterals[1], patch_size)
            laterals[1] = self.CrossAtten_p3_p2(
                p2, self.fpn_upsample(laterals[2]), patch_size)
            semantic_logits.append(sem)
            gate_maps.append(gate)

            p1, sem, gate = self.SelfAttn_p1(laterals[0], patch_size)
            laterals[0] = self.CrossAtten_p2_p1(
                p1, self.fpn_upsample(laterals[1]), patch_size)
            semantic_logits.append(sem)
            gate_maps.append(gate)
        elif used_backbone_levels == 3:
            _, _, h, w = laterals[2].size()
            patch_size = [h, w]

            laterals[2], sem, gate = self.SelfAttn_p4(
                laterals[2], patch_size)
            semantic_logits.append(sem)
            gate_maps.append(gate)

            p2, sem, gate = self.SelfAttn_p3(laterals[1], patch_size)
            laterals[1] = self.CrossAtten_p4_p3(
                p2, self.fpn_upsample(laterals[2]), patch_size)
            semantic_logits.append(sem)
            gate_maps.append(gate)

            p1, sem, gate = self.SelfAttn_p2(laterals[0], patch_size)
            laterals[0] = self.CrossAtten_p3_p2(
                p1, self.fpn_upsample(laterals[1]), patch_size)
            semantic_logits.append(sem)
            gate_maps.append(gate)
        else:
            raise AssertionError(
                'HS_SCPV1_1_FPN expects 3 or 4 input feature levels, '
                f'but got {used_backbone_levels}.')

        for i in range(used_backbone_levels - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for _ in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        outs = tuple(outs)
        if self.training and self.return_semantic_logits:
            return outs, semantic_logits, gate_maps
        return outs
