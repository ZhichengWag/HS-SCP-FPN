# ------------------------------------------------------------------------- #
# Dense HS-SCPV1-5-FPN ablation.
#
# This variant changes only the SCP feature extractor: semantic logits are
# predicted directly from each lateral feature without the fixed 8x8 spatial
# pooling used by LowFreqExtractorNoDCT.  HFP, deformable semantic attention,
# position bias, gate supervision and cross-level fusion stay unchanged.
# ------------------------------------------------------------------------- #

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from .hs_scp_fpn import HFP, LightweightSemanticHead
from .hs_scpv1_5_fpn import (
    HS_SCPV1_5_FPN,
    PositionAwareForegroundGatedDeformableSemanticAttention)

__all__ = [
    'HS_SCPV1_5_DENSE_FPN', 'HFP_SCPV1_5_Dense', 'DenseSCP',
    'DenseSemanticExtractor'
]


class DenseSemanticExtractor(BaseModule):
    """Compress an unpooled lateral feature for semantic prediction."""

    def __init__(self,
                 in_channels,
                 compress_ratio=4,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg)
        self.compress = ConvModule(
            in_channels,
            in_channels // compress_ratio,
            kernel_size=1,
            bias=False)

    def forward(self, x):
        return self.compress(x)


class DenseSCP(BaseModule):
    """SCP semantic head operating directly on the lateral feature map."""

    def __init__(self,
                 in_channels,
                 num_classes=8,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg)
        compressed_channels = in_channels // 4
        self.extractor = DenseSemanticExtractor(in_channels)
        self.semantic_head = LightweightSemanticHead(
            compressed_channels, num_classes=num_classes)

    def forward(self, x):
        return self.semantic_head(self.extractor(x))


class HFP_SCPV1_5_Dense(nn.Module):
    """HFP plus dense SCP and V1.5 position-aware semantic attention."""

    def __init__(self,
                 in_channels,
                 ratio,
                 num_classes=8,
                 patch=(8, 8),
                 attn_dim=64,
                 num_points=9,
                 isdct=True,
                 gate_init_bias=-2.0,
                 pos_dim=32,
                 pos_temperature=10000.0,
                 invalid_sample_mask=True):
        super().__init__()
        self.hfp = HFP(
            in_channels, ratio=ratio, patch=patch, isdct=isdct)
        self.scp = DenseSCP(
            in_channels, num_classes=num_classes)
        self.cross_attn = (
            PositionAwareForegroundGatedDeformableSemanticAttention(
                feat_channels=in_channels,
                num_classes=num_classes,
                attn_dim=attn_dim,
                num_points=num_points,
                gate_init_bias=gate_init_bias,
                pos_dim=pos_dim,
                pos_temperature=pos_temperature,
                invalid_sample_mask=invalid_sample_mask))

    def forward(self, x, patch_size):
        hfp_out = self.hfp(x)
        semantic_logits = self.scp(x)
        fused, gate = self.cross_attn(
            hfp_out, semantic_logits, patch_size)
        return fused, semantic_logits, gate


@MODELS.register_module()
class HS_SCPV1_5_DENSE_FPN(HS_SCPV1_5_FPN):
    """HS-SCPV1.5 with unpooled semantic prediction at every level."""

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

        common = dict(
            in_channels=out_channels,
            num_classes=num_semantic_classes,
            attn_dim=scp_attn_dim,
            num_points=scp_deform_points,
            gate_init_bias=gate_init_bias,
            pos_dim=scp_pos_dim,
            pos_temperature=scp_pos_temperature,
            invalid_sample_mask=scp_invalid_sample_mask)
        self.SelfAttn_p4 = HFP_SCPV1_5_Dense(
            ratio=None, patch=(8, 8), isdct=False, **common)
        self.SelfAttn_p3 = HFP_SCPV1_5_Dense(
            ratio=None, patch=(8, 8), isdct=False, **common)
        self.SelfAttn_p2 = HFP_SCPV1_5_Dense(
            ratio=ratio, patch=(8, 8), isdct=True, **common)
        self.SelfAttn_p1 = HFP_SCPV1_5_Dense(
            ratio=ratio, patch=(16, 16), isdct=True, **common)
