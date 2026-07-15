# ------------------------------------------------------------------------- #
# HS-SCP-FPN: HS-FPN with a Semantic Context Prior low-frequency branch.
#
# This file intentionally does not modify hs_fpn.py.  It keeps the original
# HFP/SDP structure and adds:
#   - LowFreqExtractor + LightweightSemanticHead (SCP branch)
#   - SemanticCrossAttention (HFP queries SCP)
#   - HS_SCP_FPN neck registered as a separate module
# ------------------------------------------------------------------------- #

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as DCT
from einops import rearrange
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet.registry import MODELS

__all__ = [
    'HS_SCP_FPN', 'HFP_SCP', 'SCP', 'LowFreqExtractor',
    'LowFreqExtractorNoDCT', 'LightweightSemanticHead',
    'SemanticCrossAttention'
]


def _make_group_norm(num_channels, preferred_groups=32):
    groups = min(preferred_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


# ------------------------------------------------------------------------- #
# High Frequency Perception branch, copied into this new module so the
# original hs_fpn.py remains untouched.
# ------------------------------------------------------------------------- #
class DctSpatialInteraction(BaseModule):

    def __init__(self,
                 in_channels,
                 ratio,
                 isdct=True,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d',
                     distribution='uniform')):
        super().__init__(init_cfg)
        self.ratio = ratio
        self.isdct = isdct
        if not self.isdct:
            self.spatial1x1 = nn.Sequential(
                ConvModule(in_channels, 1, kernel_size=1, bias=False))

    def forward(self, x):
        _, _, h, w = x.size()
        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            freq = DCT.dct_2d(x.float(), norm='ortho')
            weight = self._compute_weight(h, w, self.ratio).to(
                device=x.device, dtype=freq.dtype)
            weight = weight.view(1, 1, h, w).expand_as(freq)
            high = DCT.idct_2d(freq * weight, norm='ortho')
            high = high.clamp(-1e4, 1e4).to(dtype=x.dtype)
        return x * high

    @staticmethod
    def _compute_weight(h, w, ratio):
        h0 = max(1, int(h * ratio[0]))
        w0 = max(1, int(w * ratio[1]))
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight


class DctChannelInteraction(BaseModule):

    def __init__(self,
                 in_channels,
                 patch,
                 ratio,
                 isdct=True,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d',
                     distribution='uniform')):
        super().__init__(init_cfg)
        self.h = patch[0]
        self.w = patch[1]
        self.ratio = ratio
        self.isdct = isdct
        self.channel1x1 = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels,
                kernel_size=1,
                groups=32,
                bias=False))
        self.channel2x1 = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels,
                kernel_size=1,
                groups=32,
                bias=False))
        self.relu = nn.ReLU()

    def forward(self, x):
        n, c, h, w = x.size()
        if not self.isdct:
            amaxp = F.adaptive_max_pool2d(x, output_size=(1, 1))
            aavgp = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(
                self.relu(aavgp))
            return x * torch.sigmoid(self.channel2x1(channel))

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            freq = DCT.dct_2d(x.float(), norm='ortho')
            weight = self._compute_weight(h, w, self.ratio).to(
                device=x.device, dtype=freq.dtype)
            weight = weight.view(1, 1, h, w).expand_as(freq)
            high = DCT.idct_2d(freq * weight, norm='ortho')
            high = high.clamp(-1e4, 1e4).to(dtype=x.dtype)

        amaxp = F.adaptive_max_pool2d(high, output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(high, output_size=(self.h, self.w))
        amaxp = torch.sum(self.relu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2, 3]).view(n, c, 1, 1)
        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)
        return x * torch.sigmoid(self.channel2x1(channel))

    @staticmethod
    def _compute_weight(h, w, ratio):
        h0 = max(1, int(h * ratio[0]))
        w0 = max(1, int(w * ratio[1]))
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight


class HFP(BaseModule):

    def __init__(self,
                 in_channels,
                 ratio,
                 patch=(8, 8),
                 isdct=True,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d',
                     distribution='uniform')):
        super().__init__(init_cfg)
        self.spatial = DctSpatialInteraction(
            in_channels, ratio=ratio, isdct=isdct)
        self.channel = DctChannelInteraction(
            in_channels, patch=patch, ratio=ratio, isdct=isdct)
        self.out = nn.Sequential(
            ConvModule(in_channels, in_channels, kernel_size=3, padding=1),
            _make_group_norm(in_channels, 32))

    def forward(self, x):
        spatial = self.spatial(x)
        channel = self.channel(x)
        return self.out(spatial + channel)


class SDP(BaseModule):

    def __init__(self,
                 dim=256,
                 inter_dim=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d',
                     distribution='uniform')):
        super().__init__(init_cfg)
        self.inter_dim = inter_dim if inter_dim is not None else dim
        self.conv_q = nn.Sequential(
            ConvModule(dim, self.inter_dim, 1, padding=0, bias=False),
            _make_group_norm(self.inter_dim, 32))
        self.conv_k = nn.Sequential(
            ConvModule(dim, self.inter_dim, 1, padding=0, bias=False),
            _make_group_norm(self.inter_dim, 32))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_low, x_high, patch_size):
        _, _, h, w = x_low.size()
        ph, pw = patch_size
        q = rearrange(
            self.conv_q(x_low),
            'b c (h p1) (w p2) -> (b h w) c (p1 p2)',
            p1=ph,
            p2=pw).transpose(1, 2)
        k = rearrange(
            self.conv_k(x_high),
            'b c (h p1) (w p2) -> (b h w) c (p1 p2)',
            p1=ph,
            p2=pw)
        attn = torch.matmul(q, k) / np.power(self.inter_dim, 0.5)
        attn = self.softmax(attn)
        output = torch.matmul(attn, k.transpose(1, 2))
        output = rearrange(
            output.transpose(1, 2).contiguous(),
            '(b h w) c (p1 p2) -> b c (h p1) (w p2)',
            p1=ph,
            p2=pw,
            h=h // ph,
            w=w // pw)
        return output + x_low


# ------------------------------------------------------------------------- #
# SCP branch.
# ------------------------------------------------------------------------- #
class LowFreqExtractor(BaseModule):

    def __init__(self,
                 in_channels,
                 ratio=(0.25, 0.25),
                 compress_ratio=4,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d',
                     distribution='uniform')):
        super().__init__(init_cfg)
        self.ratio = ratio
        self.norm = nn.LayerNorm(in_channels)
        self.compress = ConvModule(
            in_channels, in_channels // compress_ratio, 1, bias=False)

    def forward(self, x):
        _, _, h, w = x.size()
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            freq = DCT.dct_2d(x.float(), norm='ortho')
            weight = self._compute_low_pass_weight(h, w, self.ratio).to(
                device=x.device, dtype=freq.dtype)
            weight = weight.view(1, 1, h, w).expand_as(freq)
            x_low = DCT.idct_2d(freq * weight, norm='ortho')
            x_low = x_low.clamp(-1e4, 1e4).to(dtype=x.dtype)
        x_low = self.norm(x_low.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.compress(x_low)

    @staticmethod
    def _compute_low_pass_weight(h, w, ratio):
        h0 = max(1, int(h * ratio[0]))
        w0 = max(1, int(w * ratio[1]))
        weight = torch.zeros((h, w), requires_grad=False)
        weight[:h0, :w0] = 1
        return weight


class LowFreqExtractorNoDCT(BaseModule):

    def __init__(self,
                 in_channels,
                 compress_ratio=4,
                 pool_size=(8, 8),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d',
                     distribution='uniform')):
        super().__init__(init_cfg)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.compress = ConvModule(
            in_channels, in_channels // compress_ratio, 1, bias=False)

    def forward(self, x):
        _, _, h, w = x.size()
        z = self.pool(x)
        z = F.interpolate(
            z, size=(h, w), mode='bilinear', align_corners=False)
        return self.compress(z)


class LightweightSemanticHead(BaseModule):

    def __init__(self,
                 in_channels,
                 num_classes=8,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d',
                     distribution='uniform')):
        super().__init__(init_cfg)
        ch = in_channels
        self.head = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
            nn.Conv2d(ch, ch, 1, bias=False),
            _make_group_norm(ch, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
            nn.Conv2d(ch, ch, 1, bias=False),
            _make_group_norm(ch, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, num_classes, 1))

    def forward(self, z):
        return self.head(z)


class SCP(BaseModule):

    def __init__(self,
                 in_channels,
                 ratio=(0.25, 0.25),
                 num_classes=8,
                 isdct=True,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d',
                     distribution='uniform')):
        super().__init__(init_cfg)
        compressed_ch = in_channels // 4
        if isdct:
            self.lfe = LowFreqExtractor(in_channels, ratio=ratio)
        else:
            self.lfe = LowFreqExtractorNoDCT(in_channels)
        self.semantic_head = LightweightSemanticHead(
            compressed_ch, num_classes=num_classes)

    def forward(self, x):
        z = self.lfe(x)
        return self.semantic_head(z)


class SemanticCrossAttention(BaseModule):

    def __init__(self,
                 feat_channels=256,
                 num_classes=8,
                 attn_dim=64,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d',
                     distribution='uniform')):
        super().__init__(init_cfg)
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
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hfp_feat, scp_feat, patch_size):
        _, _, h, w = hfp_feat.shape
        ph, pw = patch_size

        q = self.conv_q(hfp_feat)
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
        return hfp_feat + self.out_proj(out)


class HFP_SCP(BaseModule):

    def __init__(self,
                 in_channels,
                 ratio,
                 num_classes=8,
                 patch=(8, 8),
                 attn_dim=64,
                 isdct=True,
                 scp_isdct=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d',
                     distribution='uniform')):
        super().__init__(init_cfg)
        scp_isdct = isdct if scp_isdct is None else scp_isdct
        self.hfp = HFP(
            in_channels, ratio=ratio, patch=patch, isdct=isdct)
        self.scp = SCP(
            in_channels,
            ratio=ratio,
            num_classes=num_classes,
            isdct=scp_isdct)
        self.cross_attn = SemanticCrossAttention(
            feat_channels=in_channels,
            num_classes=num_classes,
            attn_dim=attn_dim)

    def forward(self, x, patch_size):
        hfp_out = self.hfp(x)
        semantic_logits = self.scp(x)
        fused = self.cross_attn(hfp_out, semantic_logits, patch_size)
        return fused, semantic_logits


@MODELS.register_module()
class HS_SCP_FPN(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 ratio=(0.25, 0.25),
                 num_semantic_classes=8,
                 scp_attn_dim=64,
                 scp_use_dct_lowpass=False,
                 return_semantic_logits=True,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d',
                     distribution='uniform')):
        super().__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.return_semantic_logits = return_semantic_logits
        self.scp_use_dct_lowpass = scp_use_dct_lowpass

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_upsample = lambda x: F.interpolate(
            x, scale_factor=2, mode='nearest')

        self.SelfAttn_p4 = HFP_SCP(
            out_channels,
            ratio=None,
            num_classes=num_semantic_classes,
            patch=(8, 8),
            attn_dim=scp_attn_dim,
            isdct=False,
            scp_isdct=False)
        self.SelfAttn_p3 = HFP_SCP(
            out_channels,
            ratio=None,
            num_classes=num_semantic_classes,
            patch=(8, 8),
            attn_dim=scp_attn_dim,
            isdct=False,
            scp_isdct=False)
        self.SelfAttn_p2 = HFP_SCP(
            out_channels,
            ratio=ratio,
            num_classes=num_semantic_classes,
            patch=(8, 8),
            attn_dim=scp_attn_dim,
            isdct=True,
            scp_isdct=scp_use_dct_lowpass)
        self.SelfAttn_p1 = HFP_SCP(
            out_channels,
            ratio=ratio,
            num_classes=num_semantic_classes,
            patch=(16, 16),
            attn_dim=scp_attn_dim,
            isdct=True,
            scp_isdct=scp_use_dct_lowpass)

        self.CrossAtten_p4_p3 = SDP(dim=out_channels)
        self.CrossAtten_p3_p2 = SDP(dim=out_channels)
        self.CrossAtten_p2_p1 = SDP(dim=out_channels)

        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        used_backbone_levels = len(laterals)
        semantic_logits = []

        if used_backbone_levels == 4:
            _, _, h, w = laterals[3].size()
            patch_size = [h, w]

            laterals[3], sem = self.SelfAttn_p4(laterals[3], patch_size)
            semantic_logits.append(sem)

            p3, sem = self.SelfAttn_p3(laterals[2], patch_size)
            laterals[2] = self.CrossAtten_p4_p3(
                p3, self.fpn_upsample(laterals[3]), patch_size)
            semantic_logits.append(sem)

            p2, sem = self.SelfAttn_p2(laterals[1], patch_size)
            laterals[1] = self.CrossAtten_p3_p2(
                p2, self.fpn_upsample(laterals[2]), patch_size)
            semantic_logits.append(sem)

            p1, sem = self.SelfAttn_p1(laterals[0], patch_size)
            laterals[0] = self.CrossAtten_p2_p1(
                p1, self.fpn_upsample(laterals[1]), patch_size)
            semantic_logits.append(sem)
        elif used_backbone_levels == 3:
            _, _, h, w = laterals[2].size()
            patch_size = [h, w]

            laterals[2], sem = self.SelfAttn_p4(laterals[2], patch_size)
            semantic_logits.append(sem)

            p2, sem = self.SelfAttn_p3(laterals[1], patch_size)
            laterals[1] = self.CrossAtten_p4_p3(
                p2, self.fpn_upsample(laterals[2]), patch_size)
            semantic_logits.append(sem)

            p1, sem = self.SelfAttn_p2(laterals[0], patch_size)
            laterals[0] = self.CrossAtten_p3_p2(
                p1, self.fpn_upsample(laterals[1]), patch_size)
            semantic_logits.append(sem)
        else:
            raise AssertionError(
                'HS_SCP_FPN expects 3 or 4 input feature levels, '
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
            return outs, semantic_logits
        return outs
