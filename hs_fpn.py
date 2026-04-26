#------------------------------------------------------------------#
# Code Structure of HS-FPN (https://arxiv.org/abs/2412.10116)
# HS-FPN
# ├── HFP_SCP (HFP + Semantic Context Prior, replaces HFP)
# │   ├── HFP (High Frequency Perception Module)
# │   │   ├── DctSpatialInteraction (Spatial Path of HFP)
# │   │   └── DctChannelInteraction (Channel Path of HFP)
# │   ├── SCP (Semantic Context Prior Branch)
# │   │   ├── LowFreqExtractor / LowFreqExtractor_NoDCT
# │   │   └── LightweightSemanticHead
# │   └── SemanticCrossAttention (Q←HFP, KV←SCP)
# └── SDP (Spatial Dependency Perception Module)
#-----------------------------------------------------------------#

import math
import torch
import numpy as np
import torch.nn as nn
import torch_dct as DCT
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn import ConvModule
from mmdet.models.builder import NECKS
from mmcv.runner import BaseModule, auto_fp16

__all__ =['HS-FPN']

#------------------------------------------------------------------#
# Spatial Path of HFP
# Only p1&p2 use dct to extract high_frequency response
#------------------------------------------------------------------#
class DctSpatialInteraction(BaseModule):
    def __init__(self,
                in_channels,
                ratio,
                isdct = True,
                init_cfg=dict(
                    type='Xavier', layer='Conv2d', distribution='uniform')):
        super(DctSpatialInteraction, self).__init__(init_cfg)
        self.ratio = ratio
        self.isdct = isdct # true when in p1&p2 # false when in p3&p4
        if not self.isdct:
            self.spatial1x1 = nn.Sequential(
            *[ConvModule(in_channels, 1, kernel_size=1, bias=False)]
        )

    def forward(self, x):
        _, _, h0, w0 = x.size()
        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))
        idct = DCT.dct_2d(x, norm='ortho') 
        weight = self._compute_weight(h0, w0, self.ratio).to(x.device)
        weight = weight.view(1, h0, w0).expand_as(idct)             
        dct = idct * weight # filter out low-frequency features 
        dct_ = DCT.idct_2d(dct, norm='ortho') # generate spatial mask
        return x * dct_

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight


#------------------------------------------------------------------#
# Channel Path of HFP
# Only p1&p2 use dct to extract high_frequency response
#------------------------------------------------------------------#
class DctChannelInteraction(BaseModule):
    def __init__(self,
                in_channels, 
                patch,
                ratio,
                isdct=True,
                init_cfg=dict(
                    type='Xavier', layer='Conv2d', distribution='uniform')
                ):
        super(DctChannelInteraction, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.h = patch[0]
        self.w = patch[1]
        self.ratio = ratio
        self.isdct = isdct
        self.channel1x1 = nn.Sequential(
            *[ConvModule(in_channels, in_channels, kernel_size=1, groups=32, bias=False)],
        )
        self.channel2x1 = nn.Sequential(
            *[ConvModule(in_channels, in_channels, kernel_size=1, groups=32, bias=False)],
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        n, c, h, w = x.size()
        if not self.isdct: # true when in p1&p2 # false when in p3&p4
            amaxp = F.adaptive_max_pool2d(x,  output_size=(1, 1))
            aavgp = F.adaptive_avg_pool2d(x,  output_size=(1, 1))
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp)) # 2025 03 15 szc 
            return x * torch.sigmoid(self.channel2x1(channel))

        idct = DCT.dct_2d(x, norm='ortho')
        weight = self._compute_weight(h, w, self.ratio).to(x.device)
        weight = weight.view(1, h, w).expand_as(idct)             
        dct = idct * weight # filter out low-frequency features 
        dct_ = DCT.idct_2d(dct, norm='ortho') 

        amaxp = F.adaptive_max_pool2d(dct_,  output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(dct_,  output_size=(self.h, self.w))       
        amaxp = torch.sum(self.relu(amaxp), dim=[2,3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2,3]).view(n, c, 1, 1)

        # channel = torch.cat([self.channel1x1(aavgp), self.channel1x1(amaxp)], dim = 1) # TODO: The values of aavgp and amaxp appear to be on different scales. Add is a better choice instead of concate.
        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp) # 2025 03 15 szc 
        return x * torch.sigmoid(self.channel2x1(channel))
        
    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight  


#------------------------------------------------------------------#
# High Frequency Perception Module HFP
#------------------------------------------------------------------#
class HFP(BaseModule):
    def __init__(self, 
                in_channels,
                ratio,
                patch = (8,8),
                isdct = True,
                init_cfg=dict(
                    type='Xavier', layer='Conv2d', distribution='uniform')):
        super(HFP, self).__init__(init_cfg)
        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct = isdct) 
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct = isdct)
        self.out =  nn.Sequential(
            *[ConvModule(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels)]
            )
    def forward(self, x):
        spatial = self.spatial(x) # output of spatial path
        channel = self.channel(x) # output of channel path
        return self.out(spatial + channel)


#------------------------------------------------------------------#
# Spatial Dependency Perception Module SDP
#------------------------------------------------------------------#
class SDP(BaseModule):
    def __init__(self,
                dim=256,
                inter_dim=None,
                init_cfg=dict(
                    type='Xavier', layer='Conv2d', distribution='uniform')):
        super(SDP, self).__init__(init_cfg)
        self.inter_dim=inter_dim
        if self.inter_dim == None:
            self.inter_dim = dim
        self.conv_q = nn.Sequential(*[ConvModule(dim, self.inter_dim, 1, padding=0, bias=False), nn.GroupNorm(32,self.inter_dim)])
        self.conv_k = nn.Sequential(*[ConvModule(dim, self.inter_dim, 1, padding=0, bias=False), nn.GroupNorm(32,self.inter_dim)])
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x_low, x_high, patch_size):
        b_, _, h_, w_ = x_low.size()
        q = rearrange(self.conv_q(x_low), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=patch_size[0], p2=patch_size[1])
        q = q.transpose(1,2) # 1,4096,128
        k = rearrange(self.conv_k(x_high), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=patch_size[0], p2=patch_size[1])
        attn = torch.matmul(q, k) # 1, 4096, 1024
        attn = attn / np.power(self.inter_dim, 0.5)
        attn = self.softmax(attn)
        v = k.transpose(1,2)# 1, 1024, 128
        output = torch.matmul(attn,v)# 1, 4096, 128
        output = rearrange(output.transpose(1, 2).contiguous(), '(b h w) c (p1 p2) -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=h_//patch_size[0], w=w_//patch_size[1])
        return output + x_low

#------------------------------------------------------------------#
# Improved Version of Spatial Dependency Perception Module SDP
# 2025 03 15 szc 
#------------------------------------------------------------------#
class SDP_Improved(BaseModule):
    def __init__(self,
                dim=256,
                inter_dim=None,
                init_cfg=dict(
                    type='Xavier', layer='Conv2d', distribution='uniform')):
        super(SDP_Improved, self).__init__(init_cfg)
        self.inter_dim=inter_dim
        if self.inter_dim == None:
            self.inter_dim = dim
        self.conv_q = nn.Sequential(*[ConvModule(dim, self.inter_dim, 3, padding=1, bias=False), nn.GroupNorm(32,self.inter_dim)])
        self.conv_k = nn.Sequential(*[ConvModule(dim, self.inter_dim, 3, padding=1, bias=False), nn.GroupNorm(32,self.inter_dim)])
        self.conv = nn.Sequential(*[ConvModule(self.inter_dim, dim, 3, padding=1, bias=False), nn.GroupNorm(32, dim)])
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x_low, x_high, patch_size):
        b_, _, h_, w_ = x_low.size()
        q = rearrange(self.conv_q(x_low), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=patch_size[0], p2=patch_size[1])
        q = q.transpose(1,2) # 1,4096,128
        k = rearrange(self.conv_k(x_high), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=patch_size[0], p2=patch_size[1])
        attn = torch.matmul(q, k) # 1, 4096, 1024
        attn = attn / np.power(self.inter_dim, 0.5)
        attn = self.softmax(attn)
        v = k.transpose(1,2)# 1, 1024, 128
        output = torch.matmul(attn,v)# 1, 4096, 128
        output = rearrange(output.transpose(1, 2).contiguous(), '(b h w) c (p1 p2) -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=h_//patch_size[0], w=w_//patch_size[1])
        output = self.conv(output + x_low)
        return output


#------------------------------------------------------------------#
# Low-Frequency Extractor (LFE) — DCT version for P1 & P2
#------------------------------------------------------------------#
class LowFreqExtractor(BaseModule):
    """Extract low-frequency features via DCT low-pass filtering.
    Used for P1 & P2 levels where resolution is high enough for DCT.
    """
    def __init__(self, in_channels, ratio,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(LowFreqExtractor, self).__init__(init_cfg)
        self.ratio = ratio
        self.norm = nn.LayerNorm(in_channels)
        self.compress = ConvModule(in_channels, in_channels // 4, 1, bias=False)

    def forward(self, x):
        _, _, h, w = x.size()
        x_float = x.float()                                       # FP32 protection
        X = DCT.dct_2d(x_float, norm='ortho')                     # DCT transform
        M = self._compute_low_pass_weight(h, w, self.ratio).to(x.device)
        M = M.view(1, 1, h, w).expand_as(X)
        x_low = DCT.idct_2d(X * M, norm='ortho')                  # IDCT transform
        x_low = x_low.clamp(-1e4, 1e4).to(x.dtype)                # clamp protection
        x_low = self.norm(x_low.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # LayerNorm
        return self.compress(x_low)                                 # C → C//4

    def _compute_low_pass_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.zeros((h, w), requires_grad=False)
        weight[:h0, :w0] = 1                                       # keep low-frequency
        return weight


#------------------------------------------------------------------#
# Low-Frequency Extractor (LFE) — Pool version for P3 & P4
#------------------------------------------------------------------#
class LowFreqExtractor_NoDCT(BaseModule):
    """Extract low-frequency features via average pooling.
    Used for P3 & P4 levels where resolution is too low for DCT.
    """
    def __init__(self, in_channels,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(LowFreqExtractor_NoDCT, self).__init__(init_cfg)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.compress = ConvModule(in_channels, in_channels // 4, 1, bias=False)

    def forward(self, x):
        _, _, h, w = x.size()
        z = self.pool(x)
        z = F.interpolate(z, size=(h, w), mode='bilinear', align_corners=False)
        return self.compress(z)


#------------------------------------------------------------------#
# Lightweight Semantic Head (LSH)
#------------------------------------------------------------------#
class LightweightSemanticHead(BaseModule):
    """Two-layer depthwise-separable conv + 1×1 classifier.
    Outputs K-class semantic logits.
    """
    def __init__(self, in_channels, num_classes=8,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(LightweightSemanticHead, self).__init__(init_cfg)
        ch = in_channels  # C//4
        self.head = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),   # DWConv
            nn.Conv2d(ch, ch, 1, bias=False),                         # Pointwise
            nn.GroupNorm(16, ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),   # DWConv
            nn.Conv2d(ch, ch, 1, bias=False),                         # Pointwise
            nn.GroupNorm(16, ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, num_classes, 1)                             # classifier
        )

    def forward(self, z):
        return self.head(z)  # (B, K, H, W)


#------------------------------------------------------------------#
# SCP — Semantic Context Prior Branch
#------------------------------------------------------------------#
class SCP(BaseModule):
    """Semantic Context Prior: LFE → LSH.
    Extracts low-frequency features and produces K-class semantic map.
    """
    def __init__(self, in_channels, ratio, num_classes=8, isdct=True,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(SCP, self).__init__(init_cfg)
        compressed_ch = in_channels // 4
        if isdct:
            self.lfe = LowFreqExtractor(in_channels, ratio)
        else:
            self.lfe = LowFreqExtractor_NoDCT(in_channels)
        self.semantic_head = LightweightSemanticHead(compressed_ch, num_classes)

    def forward(self, x):
        z = self.lfe(x)                               # (B, C//4, H, W)
        semantic_logits = self.semantic_head(z)          # (B, K, H, W)
        return semantic_logits


#------------------------------------------------------------------#
# Semantic Cross Attention (Q←HFP, KV←SCP)
#------------------------------------------------------------------#
class SemanticCrossAttention(BaseModule):
    """Cross Attention between HFP features (C channels) and
    SCP semantic map (K channels). Structure mirrors SDP,
    only K/V projection input dim changes from C to K.
    """
    def __init__(self, feat_channels=256, num_classes=8, attn_dim=64,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(SemanticCrossAttention, self).__init__(init_cfg)
        self.attn_dim = attn_dim

        # Q projection: C → d
        self.conv_q = nn.Sequential(
            ConvModule(feat_channels, attn_dim, 1, padding=0, bias=False),
            nn.GroupNorm(32, attn_dim)
        )
        # K projection: K → d
        self.conv_k = nn.Sequential(
            ConvModule(num_classes, attn_dim, 1, padding=0, bias=False),
            nn.GroupNorm(32, attn_dim)
        )
        # V projection: K → d
        self.conv_v = nn.Sequential(
            ConvModule(num_classes, attn_dim, 1, padding=0, bias=False),
            nn.GroupNorm(32, attn_dim)
        )
        # Output projection: d → C
        self.out_proj = ConvModule(attn_dim, feat_channels, 1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hfp_feat, scp_feat, patch_size):
        """
        Args:
            hfp_feat:   (B, C, H, W)  HFP output
            scp_feat:   (B, K, H, W)  SCP K-class semantic map
            patch_size:  [p_h, p_w]
        Returns:
            (B, C, H, W)  fused feature
        """
        B, C, H, W = hfp_feat.shape
        ph, pw = patch_size

        # Projection
        Q = self.conv_q(hfp_feat)     # (B, d, H, W)
        K = self.conv_k(scp_feat)     # (B, d, H, W)
        V = self.conv_v(scp_feat)     # (B, d, H, W)

        # Patchify: (B, d, H, W) → (B', N_p, d)
        Q = rearrange(Q, 'b d (h p1) (w p2) -> (b h w) (p1 p2) d',
                      p1=ph, p2=pw)
        K = rearrange(K, 'b d (h p1) (w p2) -> (b h w) d (p1 p2)',
                      p1=ph, p2=pw)
        V = rearrange(V, 'b d (h p1) (w p2) -> (b h w) (p1 p2) d',
                      p1=ph, p2=pw)

        # Attention: (B', N_p, d) × (B', d, N_p) → (B', N_p, N_p)
        attn = torch.matmul(Q, K) / np.power(self.attn_dim, 0.5)
        attn = self.softmax(attn)

        # Weighted aggregation: (B', N_p, N_p) × (B', N_p, d) → (B', N_p, d)
        out = torch.matmul(attn, V)

        # Restore: (B', N_p, d) → (B, d, H, W)
        out = rearrange(out.transpose(1, 2).contiguous(),
                       '(b h w) d (p1 p2) -> b d (h p1) (w p2)',
                       p1=ph, p2=pw, h=H // ph, w=W // pw)

        # Project back to C dim + residual
        return hfp_feat + self.out_proj(out)


#------------------------------------------------------------------#
# HFP_SCP — Replaces original HFP in HS-FPN
#------------------------------------------------------------------#
class HFP_SCP(BaseModule):
    """HFP and SCP run in parallel, fused via Cross Attention.
    Drop-in replacement for HFP in HS_FPN.
    """
    def __init__(self, in_channels, ratio, num_classes=8,
                 patch=(8, 8), attn_dim=64, isdct=True,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(HFP_SCP, self).__init__(init_cfg)
        self.hfp = HFP(in_channels, ratio=ratio, patch=patch, isdct=isdct)
        self.scp = SCP(in_channels, ratio=ratio, num_classes=num_classes,
                       isdct=isdct)
        self.cross_attn = SemanticCrossAttention(
            feat_channels=in_channels,
            num_classes=num_classes,
            attn_dim=attn_dim
        )

    def forward(self, x, patch_size):
        """
        Returns:
            fused_feat:      (B, C, H, W)
            semantic_logits: (B, K, H, W)  for distillation loss during training
        """
        hfp_out = self.hfp(x)                 # (B, C, H, W)
        semantic_logits = self.scp(x)           # (B, K, H, W)

        # Cross Attention fusion:
        # HFP features query SCP semantic context
        fused = self.cross_attn(hfp_out, semantic_logits, patch_size)

        return fused, semantic_logits


#------------------------------------------------------------------#
# HS_FPN (with SCP branch integration)
#------------------------------------------------------------------#
# @NECKS.register_module()
class HS_FPN(BaseModule):
    def __init__(self,
                in_channels,
                out_channels,
                num_outs,
                ratio = (0.25, 0.25),
                num_classes=8,
                attn_dim=64,
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
                    type='Xavier', layer='Conv2d', distribution='uniform')):
        super(HS_FPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
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
        def interpolate(input):
            up_mode = 'nearest'
            return F.interpolate(input, scale_factor=2, mode='nearest', align_corners=False if up_mode == 'bilinear' else None)
        self.fpn_upsample = interpolate

        # HFP replaced by HFP_SCP for all levels
        self.SelfAttn_p4 = HFP_SCP(out_channels, ratio=None, num_classes=num_classes,
                                    patch=(8, 8), attn_dim=attn_dim, isdct=False)
        self.SelfAttn_p3 = HFP_SCP(out_channels, ratio=None, num_classes=num_classes,
                                    patch=(8, 8), attn_dim=attn_dim, isdct=False)
        self.SelfAttn_p2 = HFP_SCP(out_channels, ratio=ratio, num_classes=num_classes,
                                    patch=(8, 8), attn_dim=attn_dim, isdct=True)
        self.SelfAttn_p1 = HFP_SCP(out_channels, ratio=ratio, num_classes=num_classes,
                                    patch=(16, 16), attn_dim=attn_dim, isdct=True)

        self.CrossAtten_p4_p3 = SDP(dim=out_channels)
        self.CrossAtten_p3_p2 = SDP(dim=out_channels)
        self.CrossAtten_p2_p1 = SDP(dim=out_channels)

        # add extra conv layers (e.g., RetinaNet)
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

    @auto_fp16()
    def forward(self, inputs):
        """Forward function.
        
        Returns:
            Training:  (tuple(outs), all_semantic_logits)
            Inference: tuple(outs)
        """
        assert len(inputs) == len(self.in_channels)
        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        _, _, h, w = laterals[3].size()

        all_semantic_logits = []

        # HFP_SCP returns (fused_feat, semantic_logits)
        laterals[3], sem3 = self.SelfAttn_p4(laterals[3], [h, w])
        all_semantic_logits.append(sem3)

        laterals[2], sem2 = self.SelfAttn_p3(laterals[2], [h, w])
        laterals[2] = self.CrossAtten_p4_p3(laterals[2],
                          self.fpn_upsample(laterals[3]), [h, w])
        all_semantic_logits.append(sem2)

        laterals[1], sem1 = self.SelfAttn_p2(laterals[1], [h, w])
        laterals[1] = self.CrossAtten_p3_p2(laterals[1],
                          self.fpn_upsample(laterals[2]), [h, w])
        all_semantic_logits.append(sem1)

        laterals[0], sem0 = self.SelfAttn_p1(laterals[0], [h, w])
        laterals[0] = self.CrossAtten_p2_p1(laterals[0],
                          self.fpn_upsample(laterals[1]), [h, w])
        all_semantic_logits.append(sem0)

        used_backbone_levels = len(laterals)  
        
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
            
        # # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
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

        if self.training:
            return tuple(outs), all_semantic_logits
        return tuple(outs)