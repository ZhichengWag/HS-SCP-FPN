"""Visualize whether SCP V1/V1.x foreground gates align with GT boxes.

This script reuses the runtime attention patch from
``visualize_scp_score_suppression.py`` so the model source is not modified.
It renders gate heatmaps and top-activation masks, then writes per-level
statistics into ``manifest.json``.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MMDET_ROOT = WORKSPACE_ROOT / 'mmdetection'
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (str(WORKSPACE_ROOT), str(MMDET_ROOT), str(TOOLS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from visualize_scp_cross_attention import (  # noqa: E402
    LEVEL_ORDER, build_contact_sheet, build_gt_mask, build_inference_pipeline,
    default_aitod_img_dir, infer_gt_ann_file, iter_images, load_config,
    load_gt_index, lookup_gt_boxes, make_panel, normalize_path,
    overlay_heatmap, resize_float_to_original, resolve_device)
from visualize_scp_score_suppression import (  # noqa: E402
    clear_cached_scores, collect_scores, patch_scp_v1_attention,
    reject_scpv2_config)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize SCP V1/V1.x foreground gate accuracy.')
    parser.add_argument(
        '--config',
        default=str(WORKSPACE_ROOT / 'config_hsfpn' /
                    'cascade_rcnn_r50_aitod_scpv1_5.py'),
        help='SCP V1/V1.x config path.')
    parser.add_argument(
        '--checkpoint',
        default=str(WORKSPACE_ROOT / 'work_dirs' /
                    'cascade_rcnn_r50_aitod_scpv1_5_b1_k150_t20_epoch12' /
                    'best_coco_bbox_mAP_epoch_12.pth'),
        help='Checkpoint path.')
    parser.add_argument(
        '--img-dir',
        default=default_aitod_img_dir(),
        help='Directory to sample images from.')
    parser.add_argument(
        '--out-dir',
        default=str(WORKSPACE_ROOT / 'visualizations' /
                    'scp_v1_5_gate_accuracy'),
        help='Directory to write visualizations.')
    parser.add_argument('--sample-n', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20260630)
    parser.add_argument(
        '--device',
        default='auto',
        help='Inference device. Use "auto", "cuda:0", or "cpu".')
    parser.add_argument(
        '--levels',
        default='P1,P2,P3,P4',
        help='Comma-separated levels to render, e.g. P1,P2 or P1,P2,P3,P4.')
    parser.add_argument(
        '--top-ratio',
        type=float,
        default=0.01,
        help='Fraction of highest gate pixels used as the active mask.')
    parser.add_argument(
        '--gate-threshold',
        type=float,
        default=None,
        help='Optional absolute gate threshold. Overrides --top-ratio.')
    parser.add_argument('--alpha', type=float, default=0.48)
    parser.add_argument('--mask-alpha', type=float, default=0.55)
    parser.add_argument('--panel-size', type=int, default=320)
    parser.add_argument('--thumb-size', type=int, default=220)
    parser.add_argument(
        '--no-gt-boxes',
        action='store_true',
        help='Do not draw or score against GT boxes.')
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
    return parsed or ['P1', 'P2', 'P3', 'P4']


def finite01(array):
    array = np.asarray(array, dtype=np.float32)
    return np.nan_to_num(array, nan=0.0, posinf=1.0,
                         neginf=0.0).clip(0.0, 1.0)


def heat_basic_stats(heat):
    heat = finite01(heat)
    vals = heat.reshape(-1)
    return {
        'min': float(vals.min()) if vals.size else None,
        'max': float(vals.max()) if vals.size else None,
        'mean': float(vals.mean()) if vals.size else None,
        'p50': float(np.percentile(vals, 50)) if vals.size else None,
        'p95': float(np.percentile(vals, 95)) if vals.size else None,
        'p99': float(np.percentile(vals, 99)) if vals.size else None,
    }


def active_gate_mask(gate, top_ratio=0.01, threshold=None):
    gate = finite01(gate)
    if gate.size == 0 or float(gate.max()) <= 0.0:
        return np.zeros_like(gate, dtype=bool)

    if threshold is not None:
        return gate >= float(threshold)

    top_ratio = float(np.clip(top_ratio, 0.0, 1.0))
    if top_ratio <= 0.0:
        return np.zeros_like(gate, dtype=bool)
    if top_ratio >= 1.0:
        return gate > 0.0

    flat = gate.reshape(-1)
    k = max(1, int(math.ceil(flat.size * top_ratio)))
    kth = np.partition(flat, flat.size - k)[flat.size - k]
    return gate >= kth


def gate_accuracy_stats(gate, gt_boxes, top_ratio, threshold):
    gate = finite01(gate)
    top_mask = active_gate_mask(gate, top_ratio, threshold)
    stats = heat_basic_stats(gate)
    stats.update({
        'active_rule': ('threshold' if threshold is not None else 'top_ratio'),
        'active_threshold': float(threshold) if threshold is not None else None,
        'top_ratio': None if threshold is not None else float(top_ratio),
        'active_area_ratio': float(top_mask.mean()) if top_mask.size else None,
        'inside_gt_mean': None,
        'outside_gt_mean': None,
        'inside_minus_outside': None,
        'inside_outside_ratio': None,
        'gate_mass_inside_ratio': None,
        'gt_area_ratio': None,
        'mass_enrichment': None,
        'active_precision': None,
        'active_recall': None,
        'active_iou': None,
    })

    if not gt_boxes:
        return stats, top_mask

    height, width = gate.shape
    gt_mask = build_gt_mask((width, height), gt_boxes)
    if not gt_mask.any():
        return stats, top_mask

    bg_mask = ~gt_mask
    inside = float(gate[gt_mask].mean())
    outside = float(gate[bg_mask].mean()) if bg_mask.any() else None
    gate_sum = float(gate.sum())
    gt_area_ratio = float(gt_mask.mean())
    mass_inside = (float(gate[gt_mask].sum()) / max(gate_sum, 1e-12)
                   if gate_sum > 0.0 else 0.0)

    active_sum = int(top_mask.sum())
    active_gt = int((top_mask & gt_mask).sum())
    union = int((top_mask | gt_mask).sum())

    stats.update({
        'inside_gt_mean':
        inside,
        'outside_gt_mean':
        outside,
        'inside_minus_outside':
        None if outside is None else inside - outside,
        'inside_outside_ratio':
        None if outside is None else inside / max(outside, 1e-12),
        'gate_mass_inside_ratio':
        mass_inside,
        'gt_area_ratio':
        gt_area_ratio,
        'mass_enrichment':
        mass_inside / max(gt_area_ratio, 1e-12),
        'active_precision':
        active_gt / max(active_sum, 1),
        'active_recall':
        active_gt / max(int(gt_mask.sum()), 1),
        'active_iou':
        active_gt / max(union, 1),
    })
    return stats, top_mask


def overlay_mask(image, mask, alpha, color=(255, 36, 48)):
    base = np.asarray(image.convert('RGB'), dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    color = np.asarray(color, dtype=np.float32)
    if mask.any():
        base[mask] = base[mask] * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def make_stats_panel(level_payloads, panel_size, font):
    title_h = 24
    panel = Image.new('RGB', (panel_size, panel_size + title_h),
                      (245, 245, 242))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel_size, title_h), fill=(34, 34, 34))
    draw.text((8, 5), 'Gate/GT metrics', fill=(245, 245, 245), font=font)

    y = title_h + 8
    if not level_payloads:
        draw.text((10, y), 'No gate maps collected', fill=(20, 20, 20),
                  font=font)
        return panel

    for payload in level_payloads:
        if y > panel.height - 58:
            break
        level = payload['level']
        stats = payload['stats']
        mean = stats['mean']
        p99 = stats['p99']
        inside = stats['inside_gt_mean']
        outside = stats['outside_gt_mean']
        enrich = stats['mass_enrichment']
        precision = stats['active_precision']
        recall = stats['active_recall']
        iou = stats['active_iou']

        draw.text((10, y), f'{level} mean {mean:.4f} p99 {p99:.4f}',
                  fill=(20, 20, 20), font=font)
        y += 15
        if inside is None or outside is None:
            draw.text((14, y), 'GT metrics unavailable',
                      fill=(20, 20, 20), font=font)
            y += 24
            continue

        draw.text((14, y), f'in {inside:.3f} out {outside:.3f} '
                  f'enrich {enrich:.2f}', fill=(20, 20, 20), font=font)
        y += 15
        draw.text((14, y), f'top P/R/IoU {precision:.2f}/'
                  f'{recall:.2f}/{iou:.2f}', fill=(20, 20, 20), font=font)
        y += 24
    return panel


def render_visualization(image_path, level_payloads, out_path, alpha,
                         mask_alpha, panel_size, gt_boxes, gt_line_width):
    original = Image.open(image_path).convert('RGB')
    font = ImageFont.load_default()

    panels = [
        make_panel(original, f'Original + GT ({len(gt_boxes)})',
                   panel_size, font, gt_boxes, gt_line_width)
    ]

    for payload in level_payloads:
        level = payload['level']
        gate = payload['gate']
        top_mask = payload['top_mask']
        stats = payload['stats']

        title = f'{level} gate m={stats["mean"]:.3f} max={stats["max"]:.2f}'
        panels.append(
            make_panel(overlay_heatmap(original, gate, alpha), title,
                       panel_size, font, gt_boxes, gt_line_width))
        panels.append(
            make_panel(overlay_mask(original, top_mask, mask_alpha),
                       f'{level} top gate pixels', panel_size, font,
                       gt_boxes, gt_line_width))

    panels.append(make_stats_panel(level_payloads, panel_size, font))

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


def add_aggregate(aggregate, level, stats):
    agg = aggregate.setdefault(level, {'n': 0, 'sums': {}})
    agg['n'] += 1
    for key in (
            'mean', 'p95', 'p99', 'inside_gt_mean', 'outside_gt_mean',
            'inside_minus_outside', 'inside_outside_ratio',
            'gate_mass_inside_ratio', 'gt_area_ratio', 'mass_enrichment',
            'active_area_ratio', 'active_precision', 'active_recall',
            'active_iou'):
        value = stats.get(key)
        if value is None:
            continue
        slot = agg['sums'].setdefault(key, {'sum': 0.0, 'n': 0})
        slot['sum'] += float(value)
        slot['n'] += 1


def summarize_aggregate(aggregate):
    summary = {}
    for level, agg in sorted(
            aggregate.items(),
            key=lambda item: LEVEL_ORDER.get(item[0], 99)):
        level_summary = {'images': agg['n']}
        for key, slot in agg['sums'].items():
            level_summary[key] = slot['sum'] / max(slot['n'], 1)
        summary[level] = level_summary
    return summary


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
    patched = patch_scp_v1_attention(model, attn_reduce='mean')
    if not patched:
        raise RuntimeError('No SCP V1/V1.x attention modules were found.')

    rendered = []
    manifest_items = []
    aggregate = {}

    for idx, image_path in enumerate(sampled, start=1):
        gt_boxes = lookup_gt_boxes(gt_index, image_path)
        with Image.open(image_path) as img:
            original_size = img.size

        clear_cached_scores(patched)
        with torch.no_grad():
            result = inference_detector(model, str(image_path), test_pipeline)

        items = collect_scores(model, levels, score_kind='effective')
        if not items:
            print(f'[{idx:03d}/{sample_n}] skipped, no requested levels: '
                  f'{image_path}')
            continue

        level_payloads = []
        for item in items:
            gate = resize_float_to_original(item['gate'], result.metainfo,
                                            item['level'], original_size)
            gate = finite01(gate)
            stats, top_mask = gate_accuracy_stats(
                gate, gt_boxes, args.top_ratio, args.gate_threshold)
            add_aggregate(aggregate, item['level'], stats)
            level_payloads.append({
                'name': item['name'],
                'level': item['level'],
                'attention_kind': item['kind'],
                'gate_shape': list(np.asarray(item['gate']).shape),
                'gate': gate,
                'top_mask': top_mask,
                'stats': stats,
            })

        safe_stem = image_path.stem.replace(' ', '_')
        out_path = out_dir / f'{idx:03d}_{safe_stem}_scp_gate_accuracy.png'
        render_visualization(image_path, level_payloads, out_path,
                             args.alpha, args.mask_alpha, args.panel_size,
                             gt_boxes, args.gt_line_width)
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
            'levels': [{
                'name': payload['name'],
                'level': payload['level'],
                'attention_kind': payload['attention_kind'],
                'gate_shape': payload['gate_shape'],
                'stats': payload['stats'],
            } for payload in level_payloads],
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
        'top_ratio': args.top_ratio,
        'gate_threshold': args.gate_threshold,
        'gt_ann_file': str(gt_ann_file) if gt_ann_file else None,
        'gt_boxes_used': not args.no_gt_boxes and bool(gt_index),
        'aggregate_summary': summarize_aggregate(aggregate),
        'contact_sheet': str(contact_sheet),
        'metric_notes': {
            'inside_outside_ratio':
            'Mean gate inside GT boxes divided by mean gate outside GT boxes.',
            'mass_enrichment':
            'Gate mass inside GT boxes divided by GT area ratio. Values > 1 '
            'mean gate mass is more concentrated on GT than random area.',
            'active_precision':
            'Fraction of top gate pixels that fall inside GT boxes.',
            'active_recall':
            'Fraction of GT area covered by top gate pixels.',
            'active_iou':
            'IoU between top gate pixels and the GT-box mask.',
        },
        'items': manifest_items,
    }
    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f'Wrote {len(rendered)} visualizations to {out_dir}')
    print(f'Contact sheet: {contact_sheet}')
    print(f'Manifest: {out_dir / "manifest.json"}')


if __name__ == '__main__':
    main()
