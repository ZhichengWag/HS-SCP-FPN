"""Compare raw SCP semantic scores with current V1 attention scores.

This is a focused companion to ``visualize_scp_cross_attention.py``.  It
ignores SCPV2 and monkey-patches only SCP V1/V1.x cross-attention modules at
runtime.  For each selected image and FPN level it renders:

  1. raw SCP semantic confidence, max_c softmax(SCP logits)
  2. current attention score on SCP key positions
  3. gate-weighted effective attention score
  4. SCP-minus-attention suppression map
  5. attention-minus-SCP leakage map

The default "current attention" map is gate-weighted effective attention,
because it better reflects how much semantic information is actually injected
after foreground gating.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MMDET_ROOT = WORKSPACE_ROOT / 'mmdetection'
for path in (str(WORKSPACE_ROOT), str(MMDET_ROOT), str(Path(__file__).parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

from visualize_scp_cross_attention import (  # noqa: E402
    LEVEL_ORDER, build_contact_sheet, build_gt_mask, build_inference_pipeline,
    default_aitod_img_dir, infer_gt_ann_file, infer_level, iter_images,
    load_config, load_gt_index, lookup_gt_boxes, make_panel,
    normalize_heatmap, normalize_path, overlay_heatmap,
    resize_heat_to_original, resolve_device)


WINDOW_ATTN_CLASSES = {
    'SemanticCrossAttention',
    'ForegroundGatedSemanticCrossAttention',
    'ForegroundGatedSemanticCrossAttentionV1_2',
}

DEFORM_ATTN_CLASSES = {
    'ForegroundGatedDeformableSemanticAttention',
    'SoftForegroundGatedDeformableSemanticAttention',
    'PositionAwareForegroundGatedDeformableSemanticAttention',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize SCP raw scores vs current V1 attention scores.')
    parser.add_argument(
        '--config',
        default=str(WORKSPACE_ROOT / 'config_hsfpn' /
                    'cascade_rcnn_r50_aitod_scpv1_5.py'),
        help='SCP V1/V1.x config path. SCPV2 is intentionally unsupported.')
    parser.add_argument(
        '--checkpoint',
        default=str(WORKSPACE_ROOT / 'work_dirs' /
                    'cascade_rcnn_r50_aitod_scpv1_5_b1_k150_t20_epoch12' /
                    'best_coco_bbox_mAP_epoch_12.pth'),
        help='Checkpoint path.')
    parser.add_argument(
        '--img-dir',
        default=default_aitod_img_dir(),
        help='Directory to randomly sample images from.')
    parser.add_argument(
        '--out-dir',
        default=str(WORKSPACE_ROOT / 'visualizations' /
                    'scp_score_suppression_v1_5'),
        help='Directory to write visualizations.')
    parser.add_argument('--sample-n', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20260630)
    parser.add_argument(
        '--device',
        default='auto',
        help='Inference device. Use "auto", "cuda:0", or "cpu".')
    parser.add_argument(
        '--levels',
        default='P1,P2',
        help='Comma-separated FPN levels to render, e.g. P1,P2,P3.')
    parser.add_argument(
        '--score',
        choices=('effective', 'raw'),
        default='effective',
        help='Use gate-weighted effective attention or raw softmax attention '
        'as the current attention score.')
    parser.add_argument(
        '--attn-reduce',
        choices=('mean', 'max'),
        default='mean',
        help='Reduce query-to-key attention for window attention modules.')
    parser.add_argument('--alpha', type=float, default=0.48)
    parser.add_argument('--panel-size', type=int, default=300)
    parser.add_argument('--thumb-size', type=int, default=220)
    parser.add_argument(
        '--no-gt-boxes',
        action='store_true',
        help='Do not draw ground-truth boxes from the test COCO annotation.')
    parser.add_argument(
        '--gt-line-width',
        type=int,
        default=2,
        help='Ground-truth box line width on rendered panels.')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Search images recursively under --img-dir.')
    return parser.parse_args()


def parse_levels(levels):
    parsed = [level.strip().upper() for level in str(levels).split(',')
              if level.strip()]
    return parsed or ['P1', 'P2']


def finite_clamp(x, min_val=-1e4, max_val=1e4):
    return torch.nan_to_num(
        x, nan=0.0, posinf=max_val,
        neginf=min_val).clamp(min_val, max_val)


def scp_confidence(scp_feat):
    logits = finite_clamp(scp_feat.float(), min_val=-50.0, max_val=50.0)
    prob = torch.softmax(logits, dim=1)
    score = prob.amax(dim=1, keepdim=True)
    top_class = prob.argmax(dim=1, keepdim=True)
    return score.to(dtype=scp_feat.dtype), top_class


def reduce_window_attention(attn, query_weight, bsz, gh, gw, ph, pw,
                            attn_reduce):
    """Return raw and query-weighted key-position attention heatmaps."""
    attn_float = attn.detach().float()
    if attn_reduce == 'max':
        raw_patch = attn_float.amax(dim=1)
        weighted = attn_float * query_weight.unsqueeze(-1)
        effective_patch = weighted.amax(dim=1)
    else:
        raw_patch = attn_float.mean(dim=1)
        effective_patch = (
            attn_float * query_weight.unsqueeze(-1)).mean(dim=1)

    raw_heat = raw_patch.view(bsz, gh, gw, ph, pw).permute(
        0, 1, 3, 2, 4).reshape(bsz, 1, gh * ph, gw * pw)
    effective_heat = effective_patch.view(bsz, gh, gw, ph, pw).permute(
        0, 1, 3, 2, 4).reshape(bsz, 1, gh * ph, gw * pw)
    return raw_heat, effective_heat


def project_deform_attention(coords, attn, gate):
    """Project sampled-point attention back to nearest SCP key cells."""
    bsz, num_points, _, h, w = coords.shape
    valid = ((coords[:, :, 0] >= 0) & (coords[:, :, 0] <= w - 1) &
             (coords[:, :, 1] >= 0) & (coords[:, :, 1] <= h - 1))
    x = coords[:, :, 0].round().long().clamp(0, w - 1)
    y = coords[:, :, 1].round().long().clamp(0, h - 1)
    flat = (y * w + x).view(bsz, -1)

    weights = attn.permute(0, 3, 1, 2).contiguous()  # B, K, H, W
    weights = weights * valid.to(dtype=weights.dtype)
    effective_weights = weights * gate

    raw = weights.new_zeros((bsz, h * w))
    effective = weights.new_zeros((bsz, h * w))
    raw.scatter_add_(1, flat, weights.view(bsz, -1))
    effective.scatter_add_(1, flat, effective_weights.view(bsz, -1))

    denom = max(float(h * w), 1.0)
    raw = raw.view(bsz, 1, h, w) / denom
    effective = effective.view(bsz, 1, h, w) / denom
    return raw, effective


def cache_score_payload(module, hfp_feat, scp_feat, fused, gate, raw_attn,
                        effective_attn, attn_kind, extra=None):
    with torch.no_grad():
        scp_score, top_class = scp_confidence(scp_feat)
        delta = fused - hfp_feat
        payload = {
            'name': getattr(module, '_scp_score_vis_name', ''),
            'level': getattr(module, '_scp_score_vis_level', ''),
            'kind': attn_kind,
            'scp_score': scp_score[:1].detach().float().cpu(),
            'scp_top_class': top_class[:1].detach().cpu(),
            'attn_raw': raw_attn[:1].detach().float().cpu(),
            'attn_effective': effective_attn[:1].detach().float().cpu(),
            'gate': gate[:1].detach().float().cpu(),
            'delta': delta[:1].detach().float().square().mean(
                dim=1, keepdim=True).sqrt().cpu(),
        }
        if extra:
            payload.update(extra)
        module._scp_score_vis = payload


def patch_window_attention(module, attn_reduce):

    def forward_with_capture(self, hfp_feat, scp_feat, patch_size):
        bsz, _, h, w = hfp_feat.shape
        ph, pw = int(patch_size[0]), int(patch_size[1])
        gh, gw = h // ph, w // pw

        has_gate = hasattr(self, 'query_gate')
        hfp_input = finite_clamp(hfp_feat)
        scp_input = finite_clamp(scp_feat, min_val=-50.0, max_val=50.0)
        gate = (self.query_gate(hfp_input, scp_input) if has_gate else
                hfp_input.new_ones((bsz, 1, h, w)))

        q = self.conv_q(hfp_input) * gate
        k = self.conv_k(scp_input)
        v = self.conv_v(scp_input)

        q = q.view(bsz, self.attn_dim, gh, ph, gw, pw).permute(
            0, 2, 4, 3, 5, 1).reshape(bsz * gh * gw, ph * pw,
                                       self.attn_dim)
        k = k.view(bsz, self.attn_dim, gh, ph, gw, pw).permute(
            0, 2, 4, 1, 3, 5).reshape(bsz * gh * gw, self.attn_dim,
                                      ph * pw)
        v = v.view(bsz, self.attn_dim, gh, ph, gw, pw).permute(
            0, 2, 4, 3, 5, 1).reshape(bsz * gh * gw, ph * pw,
                                       self.attn_dim)

        attn = torch.matmul(q.float(), k.float()) / math.sqrt(self.attn_dim)
        attn = finite_clamp(attn, min_val=-50.0, max_val=50.0)
        attn = self.softmax(attn).to(dtype=v.dtype)
        out = torch.matmul(attn, v)
        out = out.view(bsz, gh, gw, ph, pw, self.attn_dim).permute(
            0, 5, 1, 3, 2, 4).reshape(bsz, self.attn_dim, h, w)

        if hasattr(self, 'refine'):
            delta_eff = gate * self.out_proj(out)
            fused = hfp_input + delta_eff + self.refine(
                hfp_input, delta_eff, gate)
        elif has_gate:
            fused = hfp_input + gate * self.out_proj(out)
        else:
            fused = hfp_input + self.out_proj(out)

        with torch.no_grad():
            gate_patch = gate.detach().float().view(
                bsz, 1, gh, ph, gw, pw).permute(
                    0, 2, 4, 3, 5, 1).reshape(bsz * gh * gw, ph * pw)
            raw_heat, effective_heat = reduce_window_attention(
                attn, gate_patch, bsz, gh, gw, ph, pw, attn_reduce)
            cache_score_payload(self, hfp_input, scp_input, fused, gate,
                                raw_heat, effective_heat, 'window')
        return fused if not has_gate else (fused, gate)

    module.forward = MethodType(forward_with_capture, module)


def patch_deform_attention(module):

    def forward_with_capture(self, hfp_feat, scp_feat, patch_size=None):
        del patch_size
        bsz, _, h, w = hfp_feat.shape
        gate = self.query_gate(hfp_feat, scp_feat)
        q = self.conv_q(hfp_feat) * gate
        k = self.conv_k(scp_feat)
        v = self.conv_v(scp_feat)

        with torch.no_grad():
            semantic_conf, _ = scp_confidence(scp_feat)
        offset_input = torch.cat([hfp_feat.detach(), gate, semantic_conf], 1)
        offset = self.offset_head(offset_input)
        offset = offset.view(bsz, self.num_points, 2, h, w)
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

        pos_bias = None
        if hasattr(self, '_make_position_bias'):
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
            bsz, h, self.num_points * w, 2)

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
            bsz, self.attn_dim, h, self.num_points, w).permute(
                0, 2, 4, 3, 1)
        sampled_v = sampled_v.view(
            bsz, self.attn_dim, h, self.num_points, w).permute(
                0, 2, 4, 3, 1)
        q = q.permute(0, 2, 3, 1).unsqueeze(3)

        attn = (q * sampled_k).sum(dim=-1) / math.sqrt(self.attn_dim)
        if pos_bias is not None:
            attn = attn + pos_bias.to(dtype=attn.dtype)
        if getattr(self, 'invalid_sample_mask', False):
            valid = ((coords[:, :, 0] >= 0) & (coords[:, :, 0] <= w - 1) &
                     (coords[:, :, 1] >= 0) & (coords[:, :, 1] <= h - 1))
            valid = valid.permute(0, 2, 3, 1)
            attn = attn.masked_fill(~valid, -50.0)
        attn = torch.softmax(attn, dim=-1)
        out = (attn.unsqueeze(-1) * sampled_v).sum(dim=3)
        out = out.permute(0, 3, 1, 2).contiguous()
        delta = self.out_proj(out)
        fused = hfp_feat + gate * delta

        with torch.no_grad():
            raw_heat, effective_heat = project_deform_attention(
                coords.detach(), attn.detach().float(), gate.detach().float())
            extra = {
                'offset_abs_mean':
                float(offset.detach().float().abs().mean().cpu()),
                'offset_abs_max':
                float(offset.detach().float().abs().max().cpu()),
            }
            cache_score_payload(self, hfp_feat, scp_feat, fused, gate,
                                raw_heat, effective_heat, 'deform', extra)
        return fused, gate

    module.forward = MethodType(forward_with_capture, module)


def patch_scp_v1_attention(model, attn_reduce):
    patched = []
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        module_path = module.__class__.__module__.lower()
        if 'scpv2' in class_name.lower() or 'scpv2' in module_path:
            continue
        if class_name not in WINDOW_ATTN_CLASSES | DEFORM_ATTN_CLASSES:
            continue

        module._scp_score_vis_name = name
        module._scp_score_vis_level = infer_level(name)
        module._scp_score_vis = None
        module._scp_score_vis_original_forward = module.forward

        if class_name in WINDOW_ATTN_CLASSES:
            patch_window_attention(module, attn_reduce)
        else:
            patch_deform_attention(module)
        patched.append(module)
    return patched


def clear_cached_scores(patched_modules):
    for module in patched_modules:
        module._scp_score_vis = None


def collect_scores(model, levels, score_kind):
    keep = set(levels)
    items = []
    for _, module in model.named_modules():
        payload = getattr(module, '_scp_score_vis', None)
        if not payload:
            continue
        level = payload['level']
        if level not in keep:
            continue

        scp_score = payload['scp_score'][0, 0].numpy()
        attn_raw = payload['attn_raw'][0, 0].numpy()
        attn_effective = payload['attn_effective'][0, 0].numpy()
        current = attn_effective if score_kind == 'effective' else attn_raw
        gate = payload['gate'][0, 0].numpy()
        delta = payload['delta'][0, 0].numpy()

        raw01 = normalize_heatmap(scp_score)
        cur01 = normalize_heatmap(current)
        suppressed = np.clip(raw01 - cur01, 0.0, 1.0)
        amplified = np.clip(cur01 - raw01, 0.0, 1.0)
        items.append({
            'name': payload['name'],
            'level': level,
            'kind': payload['kind'],
            'scp_score': scp_score,
            'attn_raw': attn_raw,
            'attn_effective': attn_effective,
            'current_attn': current,
            'gate': gate,
            'delta': delta,
            'suppressed': suppressed,
            'amplified': amplified,
            'score_kind': score_kind,
            'offset_abs_mean': payload.get('offset_abs_mean'),
            'offset_abs_max': payload.get('offset_abs_max'),
        })
    items.sort(key=lambda x: LEVEL_ORDER.get(x['level'], 99))
    return items


def heat_stats(heat):
    heat = np.asarray(heat, dtype=np.float32)
    finite = np.isfinite(heat)
    if not finite.any():
        return {
            'min': None,
            'max': None,
            'mean': None,
            'p95': None,
            'norm_mean': None,
            'norm_p95': None,
        }
    vals = heat[finite]
    norm = normalize_heatmap(heat)
    norm_vals = norm[np.isfinite(norm)]
    return {
        'min': float(vals.min()),
        'max': float(vals.max()),
        'mean': float(vals.mean()),
        'p95': float(np.percentile(vals, 95)),
        'norm_mean': float(norm_vals.mean()) if norm_vals.size else None,
        'norm_p95': float(np.percentile(norm_vals, 95))
        if norm_vals.size else None,
    }


def gt_split_stats(heat, meta, level, original_size, gt_boxes):
    heat01 = normalize_heatmap(heat)
    aligned = resize_heat_to_original(heat01, meta, level, original_size)
    stats = {'inside_gt_mean': None, 'outside_gt_mean': None}
    if not gt_boxes:
        return stats
    gt_mask = build_gt_mask(original_size, gt_boxes)
    if gt_mask.any():
        stats['inside_gt_mean'] = float(aligned[gt_mask].mean())
    if (~gt_mask).any():
        stats['outside_gt_mean'] = float(aligned[~gt_mask].mean())
    return stats


def item_manifest(item, meta, original_size, gt_boxes):
    raw_gt = gt_split_stats(item['scp_score'], meta, item['level'],
                            original_size, gt_boxes)
    cur_gt = gt_split_stats(item['current_attn'], meta, item['level'],
                            original_size, gt_boxes)
    sup_gt = gt_split_stats(item['suppressed'], meta, item['level'],
                            original_size, gt_boxes)
    amp_gt = gt_split_stats(item['amplified'], meta, item['level'],
                            original_size, gt_boxes)
    return {
        'name': item['name'],
        'level': item['level'],
        'attention_kind': item['kind'],
        'current_score_kind': item['score_kind'],
        'scp_score': heat_stats(item['scp_score']),
        'attn_raw': heat_stats(item['attn_raw']),
        'attn_effective': heat_stats(item['attn_effective']),
        'current_attn': heat_stats(item['current_attn']),
        'gate': heat_stats(item['gate']),
        'delta': heat_stats(item['delta']),
        'suppressed_scp_minus_attn': heat_stats(item['suppressed']),
        'amplified_attn_minus_scp': heat_stats(item['amplified']),
        'gt_split': {
            'scp_score': raw_gt,
            'current_attn': cur_gt,
            'suppressed_scp_minus_attn': sup_gt,
            'amplified_attn_minus_scp': amp_gt,
        },
        'offset_abs_mean': item.get('offset_abs_mean'),
        'offset_abs_max': item.get('offset_abs_max'),
    }


def make_stats_panel(items, meta, original_size, gt_boxes, panel_size, font):
    title_h = 24
    panel = Image.new('RGB', (panel_size, panel_size + title_h),
                      (245, 245, 242))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel_size, title_h), fill=(34, 34, 34))
    draw.text((8, 5), 'Outside-GT score summary', fill=(245, 245, 245),
              font=font)

    y = title_h + 8
    if not gt_boxes:
        draw.text((10, y), 'No GT boxes loaded', fill=(20, 20, 20), font=font)
        return panel

    for item in items:
        if y > panel.height - 54:
            break
        level = item['level']
        raw_gt = gt_split_stats(item['scp_score'], meta, level, original_size,
                                gt_boxes)
        cur_gt = gt_split_stats(item['current_attn'], meta, level,
                                original_size, gt_boxes)
        sup_gt = gt_split_stats(item['suppressed'], meta, level,
                                original_size, gt_boxes)
        amp_gt = gt_split_stats(item['amplified'], meta, level, original_size,
                                gt_boxes)
        raw_out = raw_gt['outside_gt_mean']
        cur_out = cur_gt['outside_gt_mean']
        sup_out = sup_gt['outside_gt_mean']
        amp_out = amp_gt['outside_gt_mean']
        raw_in = raw_gt['inside_gt_mean']
        cur_in = cur_gt['inside_gt_mean']

        draw.text((10, y), f'{level} {item["kind"]}', fill=(20, 20, 20),
                  font=font)
        y += 15
        draw.text((14, y),
                  f'out raw {raw_out:.2f} attn {cur_out:.2f} '
                  f'sup {sup_out:.2f} leak {amp_out:.2f}',
                  fill=(20, 20, 20),
                  font=font)
        y += 15
        draw.text((14, y), f'in  raw {raw_in:.2f} attn {cur_in:.2f}',
                  fill=(20, 20, 20), font=font)
        y += 22
    return panel


def render_visualization(image_path, items, meta, out_path, alpha, panel_size,
                         gt_boxes, gt_line_width):
    original = Image.open(image_path).convert('RGB')
    original_size = original.size
    font = ImageFont.load_default()

    panels = [
        make_panel(original, f'Original + GT ({len(gt_boxes)})', panel_size,
                   font, gt_boxes, gt_line_width)
    ]

    for item in items:
        level = item['level']
        panel_specs = (
            ('scp_score', 'SCP raw score'),
            ('current_attn', f'{item["score_kind"]} attn'),
            ('suppressed', 'SCP - attn'),
            ('amplified', 'attn - SCP'),
            ('gate', 'foreground gate'),
            ('delta', 'SCP delta RMS'),
        )
        for key, title in panel_specs:
            heat = item[key]
            heat01 = normalize_heatmap(heat)
            heat01 = resize_heat_to_original(heat01, meta, level,
                                             original_size)
            overlay = overlay_heatmap(original, heat01, alpha)
            panels.append(
                make_panel(overlay, f'{level} {title}', panel_size, font,
                           gt_boxes, gt_line_width))

    panels.append(
        make_stats_panel(items, meta, original_size, gt_boxes, panel_size,
                         font))

    cols = 3
    rows = math.ceil(len(panels) / cols)
    title_h = 24
    canvas = Image.new('RGB', (cols * panel_size,
                               rows * (panel_size + title_h)),
                       (245, 245, 242))
    for idx, panel in enumerate(panels):
        x = (idx % cols) * panel_size
        y = (idx // cols) * (panel_size + title_h)
        canvas.paste(panel, (x, y))
    canvas.save(out_path)


def reject_scpv2_config(config_path):
    text = Path(config_path).read_text(encoding='utf-8', errors='ignore')
    lowered = text.lower()
    if 'scpv2' in lowered or 'scp_v2' in lowered or 'hs_scpv2' in lowered:
        raise RuntimeError(
            'This script intentionally ignores SCPV2. Please use an SCP V1/'
            'V1.x config such as cascade_rcnn_r50_aitod_scpv1_5.py.')


def main():
    args = parse_args()
    levels = parse_levels(args.levels)
    config_path = normalize_path(args.config)
    checkpoint_path = normalize_path(args.checkpoint)
    img_dir = normalize_path(args.img_dir)
    out_dir = normalize_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        raise FileNotFoundError(f'Config not found: {config_path}')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    if not img_dir.exists():
        raise FileNotFoundError(f'Image directory not found: {img_dir}')
    reject_scpv2_config(config_path)

    image_paths = iter_images(img_dir, recursive=args.recursive)
    if not image_paths:
        raise RuntimeError(f'No images found in {img_dir}')
    rng = random.Random(args.seed)
    sample_n = min(args.sample_n, len(image_paths))
    sampled = rng.sample(image_paths, sample_n)

    from mmdet.apis import inference_detector, init_detector

    device = resolve_device(args.device)
    cfg = load_config(str(config_path))

    gt_index = {}
    gt_ann_file = None
    if not args.no_gt_boxes:
        gt_ann_file = infer_gt_ann_file(cfg)
        if gt_ann_file and gt_ann_file.exists():
            gt_index, _ = load_gt_index(gt_ann_file)
            print(f'Loaded GT boxes from {gt_ann_file}')
        else:
            print('GT boxes disabled: annotation file was not found from '
                  'the test dataloader config.')

    model = init_detector(cfg, str(checkpoint_path), device=device)
    test_pipeline = build_inference_pipeline(model.cfg)
    patched = patch_scp_v1_attention(model, args.attn_reduce)
    if not patched:
        raise RuntimeError(
            'No SCP V1/V1.x attention modules were found. SCPV2 is ignored.')

    rendered = []
    manifest_items = []
    for idx, image_path in enumerate(sampled, start=1):
        gt_boxes = lookup_gt_boxes(gt_index, image_path)
        with Image.open(image_path) as img:
            original_size = img.size

        clear_cached_scores(patched)
        with torch.no_grad():
            result = inference_detector(model, str(image_path), test_pipeline)

        items = collect_scores(model, levels, args.score)
        if not items:
            print(f'[{idx:03d}/{sample_n}] skipped, no requested levels: '
                  f'{image_path}')
            continue

        safe_stem = image_path.stem.replace(' ', '_')
        out_path = out_dir / f'{idx:03d}_{safe_stem}_scp_score_diff.png'
        render_visualization(image_path, items, result.metainfo, out_path,
                             args.alpha, args.panel_size, gt_boxes,
                             args.gt_line_width)
        rendered.append(out_path)

        manifest_items.append({
            'index':
            idx,
            'image':
            str(image_path),
            'output':
            str(out_path),
            'gt_box_count':
            len(gt_boxes),
            'levels': [
                item_manifest(item, result.metainfo, original_size, gt_boxes)
                for item in items
            ],
        })

        if idx == 1 or idx % 10 == 0 or idx == sample_n:
            print(f'[{idx:03d}/{sample_n}] wrote {out_path}')

    contact_sheet = out_dir / 'contact_sheet.jpg'
    build_contact_sheet(rendered, contact_sheet, args.thumb_size)
    manifest = {
        'config': str(config_path),
        'checkpoint': str(checkpoint_path),
        'img_dir': str(img_dir),
        'out_dir': str(out_dir),
        'seed': args.seed,
        'sample_n': sample_n,
        'rendered_n': len(rendered),
        'device': device,
        'levels': levels,
        'current_score_kind': args.score,
        'attn_reduce': args.attn_reduce,
        'gt_ann_file': str(gt_ann_file) if gt_ann_file else None,
        'contact_sheet': str(contact_sheet),
        'score_definitions': {
            'scp_score': 'max_c softmax(SCP semantic logits)',
            'attn_raw': 'SCP key-position attention score from softmax A',
            'attn_effective': 'attn_raw weighted by foreground query gate',
            'suppressed': 'positive part of normalized SCP - attention',
            'amplified': 'positive part of normalized attention - SCP',
        },
        'items': manifest_items,
    }
    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f'Wrote {len(rendered)} visualizations to {out_dir}')
    print(f'Contact sheet: {contact_sheet}')
    print(f'Manifest: {out_dir / "manifest.json"}')
    print('How to read: SCP raw score is the original semantic confidence. '
          'Current attention defaults to gate-weighted effective attention. '
          'Warm SCP-attn panels indicate high raw SCP response that is now '
          'suppressed; warm attn-SCP panels indicate possible attention '
          'leakage into locations not strongly supported by SCP.')


if __name__ == '__main__':
    main()
