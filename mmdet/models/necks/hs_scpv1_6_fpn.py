# ------------------------------------------------------------------------- #
# HS-SCPV1-6-FPN: SCPV1-5 with cross-level guided foreground gates on P1/P2.
#
# V1.6 keeps V1.5's position-aware deformable semantic attention.  Only the
# P1/P2 gate prediction is changed:
#   P2 gate = Gate(P2 HFP, P2 SCP confidence, up(P3 gate), up(P3 confidence))
#   P1 gate = Gate(P1 HFP, P1 SCP confidence, up(P2 gate), up(P2 confidence))
# P3/P4 remain self-gated as in V1.5.
# ------------------------------------------------------------------------- #

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet.registry import MODELS
from .hs_scp_fpn import HFP, SCP, _make_group_norm
from .hs_scpv1_1_fpn import _finite_clamp
from .hs_scpv1_5_fpn import (HS_SCPV1_5_FPN,
                             PositionAwareForegroundGatedDeformableSemanticAttention)

__all__ = [
    'HS_SCPV1_6_FPN', 'HFP_SCPV1_6', 'CrossLevelForegroundQueryGate',
    'CrossLevelPositionAwareForegroundGatedDeformableSemanticAttention'
]


def _semantic_confidence(semantic_logits, dtype=None):
    with torch.no_grad():
        semantic_logits = _finite_clamp(
            semantic_logits.float(), min_val=-50.0, max_val=50.0)
        semantic_conf = torch.softmax(
            semantic_logits, dim=1).max(dim=1, keepdim=True)[0]
        if dtype is not None:
            semantic_conf = semantic_conf.to(dtype=dtype)
    return semantic_conf


class CrossLevelForegroundQueryGate(nn.Module):
    """Foreground gate with detached guidance from the next coarser level."""

    def __init__(self, feat_channels=256, init_bias=-2.0):
        super().__init__()
        hidden_channels = max(16, feat_channels // 4)
        self.net = nn.Sequential(
            ConvModule(
                feat_channels + 3,
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

    @staticmethod
    def _prepare_guide(guide, ref, dtype):
        if guide is None:
            return ref.new_zeros(ref.shape)
        guide = guide.detach().to(device=ref.device, dtype=dtype)
        if guide.shape[-2:] != ref.shape[-2:]:
            guide = F.interpolate(
                guide,
                size=ref.shape[-2:],
                mode='bilinear',
                align_corners=False)
        return torch.nan_to_num(
            guide, nan=0.0, posinf=1.0,
            neginf=0.0).clamp(0.0, 1.0)

    def forward(self,
                hfp_feat,
                semantic_logits,
                guide_gate=None,
                guide_semantic_conf=None):
        semantic_conf = _semantic_confidence(
            semantic_logits, dtype=hfp_feat.dtype)
        gate_feat = _finite_clamp(hfp_feat.detach())
        guide_gate = self._prepare_guide(guide_gate, semantic_conf,
                                         hfp_feat.dtype)
        guide_semantic_conf = self._prepare_guide(guide_semantic_conf,
                                                  semantic_conf,
                                                  hfp_feat.dtype)
        gate_input = torch.cat(
            [gate_feat, semantic_conf, guide_gate, guide_semantic_conf],
            dim=1)
        gate = torch.sigmoid(self.net(gate_input))
        return torch.nan_to_num(
            gate, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


class CrossLevelPositionAwareForegroundGatedDeformableSemanticAttention(
        PositionAwareForegroundGatedDeformableSemanticAttention):
    """V1.5 attention with cross-level guided P1/P2 gate inputs."""

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
            gate_init_bias=gate_init_bias,
            pos_dim=pos_dim,
            pos_temperature=pos_temperature,
            invalid_sample_mask=invalid_sample_mask)
        self.query_gate = CrossLevelForegroundQueryGate(
            feat_channels=feat_channels, init_bias=gate_init_bias)

    def forward(self,
                hfp_feat,
                scp_feat,
                patch_size=None,
                guide_gate=None,
                guide_semantic_conf=None):
        del patch_size
        b, _, h, w = hfp_feat.shape
        gate = self.query_gate(hfp_feat, scp_feat, guide_gate,
                               guide_semantic_conf)
        q = self.conv_q(hfp_feat) * gate
        k = self.conv_k(scp_feat)
        v = self.conv_v(scp_feat)

        semantic_conf = _semantic_confidence(scp_feat, dtype=hfp_feat.dtype)
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

        attn = (q * sampled_k).sum(dim=-1) / self.attn_dim**0.5
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


class HFP_SCPV1_6(nn.Module):
    """HFP + SCP branch with V1.6 cross-level guided gates."""

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
        self.hfp = HFP(in_channels, ratio=ratio, patch=patch, isdct=isdct)
        self.scp = SCP(
            in_channels,
            ratio=ratio,
            num_classes=num_classes,
            isdct=scp_isdct)
        self.cross_attn = (
            CrossLevelPositionAwareForegroundGatedDeformableSemanticAttention(
                feat_channels=in_channels,
                num_classes=num_classes,
                attn_dim=attn_dim,
                num_points=num_points,
                gate_init_bias=gate_init_bias,
                pos_dim=pos_dim,
                pos_temperature=pos_temperature,
                invalid_sample_mask=invalid_sample_mask))

    def forward(self,
                x,
                patch_size,
                guide_gate=None,
                guide_semantic_conf=None):
        hfp_out = self.hfp(x)
        semantic_logits = self.scp(x)
        fused, gate = self.cross_attn(hfp_out, semantic_logits, patch_size,
                                      guide_gate, guide_semantic_conf)
        return fused, semantic_logits, gate


@MODELS.register_module()
class HS_SCPV1_6_FPN(HS_SCPV1_5_FPN):
    """HS-SCPV1-5 with cross-level guided gates on P1 and P2."""

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
            scp_pos_dim=scp_pos_dim,
            scp_pos_temperature=scp_pos_temperature,
            scp_invalid_sample_mask=scp_invalid_sample_mask,
            **kwargs)
        out_channels = self.out_channels
        ratio = kwargs.get('ratio', (0.25, 0.25))

        self.SelfAttn_p2 = HFP_SCPV1_6(
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
        self.SelfAttn_p1 = HFP_SCPV1_6(
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

            laterals[3], sem4, gate4 = self.SelfAttn_p4(
                laterals[3], patch_size)
            semantic_logits.append(sem4)
            gate_maps.append(gate4)

            p3, sem3, gate3 = self.SelfAttn_p3(laterals[2], patch_size)
            laterals[2] = self.CrossAtten_p4_p3(
                p3, self.fpn_upsample(laterals[3]), patch_size)
            semantic_logits.append(sem3)
            gate_maps.append(gate3)

            p3_conf = _semantic_confidence(sem3, dtype=laterals[1].dtype)
            p2, sem2, gate2 = self.SelfAttn_p2(
                laterals[1],
                patch_size,
                guide_gate=gate3,
                guide_semantic_conf=p3_conf)
            laterals[1] = self.CrossAtten_p3_p2(
                p2, self.fpn_upsample(laterals[2]), patch_size)
            semantic_logits.append(sem2)
            gate_maps.append(gate2)

            p2_conf = _semantic_confidence(sem2, dtype=laterals[0].dtype)
            p1, sem1, gate1 = self.SelfAttn_p1(
                laterals[0],
                patch_size,
                guide_gate=gate2,
                guide_semantic_conf=p2_conf)
            laterals[0] = self.CrossAtten_p2_p1(
                p1, self.fpn_upsample(laterals[1]), patch_size)
            semantic_logits.append(sem1)
            gate_maps.append(gate1)
        elif used_backbone_levels == 3:
            _, _, h, w = laterals[2].size()
            patch_size = [h, w]

            laterals[2], sem4, gate4 = self.SelfAttn_p4(
                laterals[2], patch_size)
            semantic_logits.append(sem4)
            gate_maps.append(gate4)

            p3, sem3, gate3 = self.SelfAttn_p3(laterals[1], patch_size)
            laterals[1] = self.CrossAtten_p4_p3(
                p3, self.fpn_upsample(laterals[2]), patch_size)
            semantic_logits.append(sem3)
            gate_maps.append(gate3)

            p3_conf = _semantic_confidence(sem3, dtype=laterals[0].dtype)
            p2, sem2, gate2 = self.SelfAttn_p2(
                laterals[0],
                patch_size,
                guide_gate=gate3,
                guide_semantic_conf=p3_conf)
            laterals[0] = self.CrossAtten_p3_p2(
                p2, self.fpn_upsample(laterals[1]), patch_size)
            semantic_logits.append(sem2)
            gate_maps.append(gate2)
        else:
            raise AssertionError(
                'HS_SCPV1_6_FPN expects 3 or 4 input feature levels, '
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
