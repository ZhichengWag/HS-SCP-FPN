"""Visualize SCPV2 high-frequency detail gates.

The script does not modify model source.  It registers runtime hooks on
``SCPV2DetailGate`` modules, renders gate heatmaps, and writes channel-ranking
statistics that help judge whether the gate emphasizes target-related channels.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MMDET_ROOT = WORKSPACE_ROOT / 'mmdetection'
for path in (str(WORKSPACE_ROOT), str(MMDET_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
LEVEL_STRIDES = {'P1': 4, 'P2': 8, 'P3': 16, 'P4': 32}
LEVEL_ORDER = {'P1': 1, 'P2': 2, 'P3': 3, 'P4': 4}

LANCZOS = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS')
BILINEAR = getattr(getattr(Image, 'Resampling', Image), 'BILINEAR')


def default_aitod_img_dir():
    if os.name == 'nt':
        return r'E:\AI-TOD\test\images'
    return '/mnt/e/AI-TOD/test/images'


def normalize_path(path):
    path = str(path)
    if os.name == 'nt':
        return Path(path)
    if len(path) >= 3 and path[1] == ':' and path[2] in ('\\', '/'):
        drive = path[0].lower()
        tail = path[3:].replace('\\', '/')
        return Path(f'/mnt/{drive}/{tail}')
    return Path(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize SCPV2 P1/P2 high-frequency detail gates.')
    parser.add_argument(
        '--config',
        default=str(WORKSPACE_ROOT / 'config_hsfpn' /
                    'cascade_rcnn_r50_aitod_scpv2.py'),
        help='HS-SCPV2-FPN config path.')
    parser.add_argument(
        '--checkpoint',
        default=str(WORKSPACE_ROOT / 'work_dirs' /
                    'cascade_rcnn_r50_aitod_scpv2_k150_0.005_epoch24' /
                    'best_coco_bbox_mAP_epoch_24.pth'),
        help='SCPV2 checkpoint path.')
    parser.add_argument(
        '--img-dir',
        default=default_aitod_img_dir(),
        help='Directory to randomly sample images from.')
    parser.add_argument(
        '--out-dir',
        default=str(WORKSPACE_ROOT / 'visualizations' /
                    'scpv2_detail_gate_100'),
        help='Directory to write gate visualizations.')
    parser.add_argument('--sample-n', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260604)
    parser.add_argument(
        '--device',
        default='auto',
        help='Inference device. Use "auto", "cuda:0", or "cpu".')
    parser.add_argument(
        '--top-k',
        type=int,
        default=8,
        help='Number of top gate channels to report per feature level.')
    parser.add_argument('--alpha', type=float, default=0.48)
    parser.add_argument('--panel-size', type=int, default=300)
    parser.add_argument('--thumb-size', type=int, default=220)
    parser.add_argument(
        '--no-gt-boxes',
        action='store_true',
        help='Do not draw or use GT boxes for channel focus statistics.')
    parser.add_argument(
        '--gt-line-width',
        type=int,
        default=2,
        help='Ground-truth box line width on the original panel.')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Search images recursively under --img-dir.')
    return parser.parse_args()


def resolve_device(device):
    if device == 'auto':
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device


def iter_images(img_dir, recursive=False):
    img_dir = Path(img_dir)
    pattern_iter = img_dir.rglob('*') if recursive else img_dir.iterdir()
    return sorted(
        path for path in pattern_iter
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def load_config(config_path):
    from mmengine.config import Config
    from mmengine.utils import import_modules_from_strings

    cfg = Config.fromfile(config_path)
    custom_imports = cfg.get('custom_imports', None)
    if custom_imports:
        import_modules_from_strings(**custom_imports)
    return cfg


def build_inference_pipeline(cfg):
    from mmcv.transforms import Compose
    from mmdet.utils import get_test_pipeline_cfg

    pipeline_cfg = get_test_pipeline_cfg(cfg)
    cleaned = []
    for transform in pipeline_cfg:
        transform_type = transform.get('type', '')
        if transform_type in ('LoadAnnotations', 'mmdet.LoadAnnotations'):
            continue
        cleaned.append(transform)
    return Compose(cleaned)


def unwrap_dataset_cfg(dataset_cfg):
    while isinstance(dataset_cfg, dict):
        if 'ann_file' in dataset_cfg:
            return dataset_cfg
        if 'dataset' in dataset_cfg:
            dataset_cfg = dataset_cfg['dataset']
            continue
        if 'datasets' in dataset_cfg and dataset_cfg['datasets']:
            dataset_cfg = dataset_cfg['datasets'][0]
            continue
        break
    return dataset_cfg


def infer_gt_ann_file(cfg):
    test_dataloader = cfg.get('test_dataloader', None)
    if test_dataloader is None:
        return None
    dataset_cfg = unwrap_dataset_cfg(test_dataloader.get('dataset', {}))
    ann_file = dataset_cfg.get('ann_file', None)
    if ann_file is None:
        return None

    ann_path = normalize_path(ann_file)
    if ann_path.is_absolute():
        return ann_path

    data_root = dataset_cfg.get('data_root', cfg.get('data_root', ''))
    if data_root:
        return normalize_path(data_root) / ann_path
    return ann_path


def load_gt_index(ann_file):
    ann_file = normalize_path(ann_file)
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    categories = {
        cat['id']: cat.get('name', str(cat['id']))
        for cat in data.get('categories', [])
    }
    image_info = {img['id']: img for img in data.get('images', [])}
    by_image_id = {}
    for ann in data.get('annotations', []):
        if ann.get('ignore', 0) or ann.get('iscrowd', 0):
            continue
        bbox = ann.get('bbox', None)
        if bbox is None or len(bbox) != 4:
            continue
        x, y, w, h = [float(v) for v in bbox]
        if w <= 0 or h <= 0:
            continue
        by_image_id.setdefault(ann['image_id'], []).append({
            'bbox': [x, y, w, h],
            'category_id': ann.get('category_id'),
            'category_name': categories.get(ann.get('category_id'), ''),
        })

    gt_index = {}
    for image_id, info in image_info.items():
        file_name = Path(info.get('file_name', '')).name
        anns = by_image_id.get(image_id, [])
        if not file_name:
            continue
        gt_index[file_name] = anns
        gt_index[Path(file_name).stem] = anns
    return gt_index, categories


def lookup_gt_boxes(gt_index, image_path):
    if not gt_index:
        return []
    return gt_index.get(image_path.name, gt_index.get(image_path.stem, []))


def infer_level(name):
    for level in ('P1', 'P2', 'P3', 'P4'):
        if f'SelfAttn_p{level[-1]}' in name:
            return level
    return name


def register_detail_gate_hooks(model):
    hooks = []
    module_map = dict(model.named_modules())

    def make_hook(name):
        parent_name = name.rsplit('.detail_gate', 1)[0]
        parent = module_map.get(parent_name)
        beta = None
        if parent is not None and hasattr(parent, 'high_to_low_beta'):
            beta = float(parent.high_to_low_beta.detach().cpu())

        def hook(module, _inputs, output):
            module._scpv2_gate_vis = {
                'name': name,
                'level': infer_level(name),
                'beta': beta,
                'gate': output[:1].detach().float().cpu(),
            }

        return hook

    for name, module in model.named_modules():
        if module.__class__.__name__ != 'SCPV2DetailGate':
            continue
        module._scpv2_gate_vis = None
        hooks.append(module.register_forward_hook(make_hook(name)))

    return hooks


def clear_cached_gates(model):
    for module in model.modules():
        if hasattr(module, '_scpv2_gate_vis'):
            module._scpv2_gate_vis = None


def collect_gates(model):
    items = []
    for module in model.modules():
        payload = getattr(module, '_scpv2_gate_vis', None)
        if not payload:
            continue
        gate = payload['gate'][0].numpy()
        items.append({
            'name': payload['name'],
            'level': payload['level'],
            'beta': payload['beta'],
            'gate': gate,
            'gate_shape': list(gate.shape),
        })
    items.sort(key=lambda x: LEVEL_ORDER.get(x['level'], 99))
    return items


def normalize_heatmap(heat):
    heat = np.asarray(heat, dtype=np.float32)
    finite = np.isfinite(heat)
    if not finite.any():
        return np.zeros_like(heat, dtype=np.float32)
    values = heat[finite]
    lo, hi = np.percentile(values, [2, 98])
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros_like(heat, dtype=np.float32)
    heat = np.clip((heat - lo) / (hi - lo), 0.0, 1.0)
    heat[~finite] = 0.0
    return heat.astype(np.float32, copy=False)


def colorize_heatmap(heat01):
    stops = np.array(
        [
            (31, 48, 94),
            (28, 145, 212),
            (253, 216, 53),
            (220, 36, 48),
        ],
        dtype=np.float32)
    x = np.clip(heat01, 0.0, 1.0) * (len(stops) - 1)
    idx = np.floor(x).astype(np.int32)
    idx = np.clip(idx, 0, len(stops) - 2)
    frac = (x - idx)[..., None]
    rgb = stops[idx] * (1.0 - frac) + stops[idx + 1] * frac
    return np.clip(rgb, 0, 255).astype(np.uint8)


def resize_heat_to_original(heat, meta, level, original_size):
    ori_w, ori_h = original_size
    img_shape = meta.get('img_shape', None)
    if img_shape is None:
        out = Image.fromarray((heat * 255).astype(np.uint8), mode='L')
        return np.asarray(out.resize((ori_w, ori_h), BILINEAR),
                          dtype=np.float32) / 255.0

    img_h, img_w = int(img_shape[0]), int(img_shape[1])
    stride = LEVEL_STRIDES.get(level)
    if stride is None:
        pad_h, pad_w = img_h, img_w
    else:
        pad_h, pad_w = heat.shape[0] * stride, heat.shape[1] * stride

    heat_img = Image.fromarray((heat * 255).astype(np.uint8), mode='L')
    heat_img = heat_img.resize((pad_w, pad_h), BILINEAR)
    heat_img = heat_img.crop((0, 0, min(img_w, pad_w), min(img_h, pad_h)))
    if heat_img.size != (img_w, img_h):
        heat_img = heat_img.resize((img_w, img_h), BILINEAR)
    heat_img = heat_img.resize((ori_w, ori_h), BILINEAR)
    return np.asarray(heat_img, dtype=np.float32) / 255.0


def overlay_heatmap(image, heat, alpha):
    heat01 = normalize_heatmap(heat)
    color = colorize_heatmap(heat01).astype(np.float32)
    base = np.asarray(image.convert('RGB'), dtype=np.float32)
    blended = base * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def build_gt_mask(size, gt_boxes):
    width, height = size
    mask_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for gt in gt_boxes:
        x, y, w, h = gt['bbox']
        draw.rectangle((x, y, x + w, y + h), fill=255)
    return np.asarray(mask_img, dtype=bool)


def channel_stats(gate, meta, level, original_size, gt_boxes, top_k):
    gate = np.asarray(gate, dtype=np.float32)
    channel_mean = gate.mean(axis=(1, 2))
    top_k = min(max(int(top_k), 1), gate.shape[0])
    top_by_mean = np.argsort(channel_mean)[::-1][:top_k]

    stats = {
        'channel_count': int(gate.shape[0]),
        'gate_min': float(gate.min()),
        'gate_max': float(gate.max()),
        'gate_mean': float(gate.mean()),
        'top_by_mean': [int(i) for i in top_by_mean],
        'top_by_gt_contrast': [],
        'top_overlap_at_k': None,
        'gt_focus_ratio_mean_top_by_mean': None,
        'channels': [],
    }
    if not gt_boxes:
        for ch in top_by_mean:
            stats['channels'].append({
                'channel': int(ch),
                'mean': float(channel_mean[ch]),
                'inside_gt_mean': None,
                'outside_gt_mean': None,
                'gt_contrast': None,
                'gt_focus_ratio': None,
            })
        return stats

    mask = build_gt_mask(original_size, gt_boxes)
    if not mask.any() or (~mask).sum() == 0:
        return stats

    inside_means = []
    outside_means = []
    contrasts = []
    ratios = []
    for ch in range(gate.shape[0]):
        heat = resize_heat_to_original(gate[ch], meta, level, original_size)
        inside = float(heat[mask].mean())
        outside = float(heat[~mask].mean())
        contrast = inside - outside
        ratio = inside / max(outside, 1e-6)
        inside_means.append(inside)
        outside_means.append(outside)
        contrasts.append(contrast)
        ratios.append(ratio)

    inside_means = np.array(inside_means, dtype=np.float32)
    outside_means = np.array(outside_means, dtype=np.float32)
    contrasts = np.array(contrasts, dtype=np.float32)
    ratios = np.array(ratios, dtype=np.float32)
    top_by_contrast = np.argsort(contrasts)[::-1][:top_k]
    overlap = len(set(top_by_mean.tolist()) & set(top_by_contrast.tolist()))

    stats['top_by_gt_contrast'] = [int(i) for i in top_by_contrast]
    stats['top_overlap_at_k'] = int(overlap)
    stats['gt_focus_ratio_mean_top_by_mean'] = float(
        ratios[top_by_mean].mean())

    reported = []
    for ch in list(top_by_mean) + [
            c for c in top_by_contrast if c not in set(top_by_mean.tolist())
    ]:
        reported.append({
            'channel': int(ch),
            'mean': float(channel_mean[ch]),
            'inside_gt_mean': float(inside_means[ch]),
            'outside_gt_mean': float(outside_means[ch]),
            'gt_contrast': float(contrasts[ch]),
            'gt_focus_ratio': float(ratios[ch]),
        })
    stats['channels'] = reported
    return stats


def fit_to_panel(image, panel_size, gt_boxes=None, gt_line_width=2):
    image = image.convert('RGB')
    src_w, src_h = image.size
    image.thumbnail((panel_size, panel_size), LANCZOS)
    canvas = Image.new('RGB', (panel_size, panel_size), (245, 245, 242))
    x = (panel_size - image.width) // 2
    y = (panel_size - image.height) // 2
    canvas.paste(image, (x, y))
    if gt_boxes:
        draw_gt_boxes(canvas, gt_boxes, (src_w, src_h), (x, y),
                      image.size, gt_line_width)
    return canvas


def draw_gt_boxes(canvas, gt_boxes, src_size, offset, draw_size, line_width):
    draw = ImageDraw.Draw(canvas)
    src_w, src_h = src_size
    dst_w, dst_h = draw_size
    off_x, off_y = offset
    scale_x = dst_w / max(src_w, 1)
    scale_y = dst_h / max(src_h, 1)
    line_width = max(1, int(line_width))
    min_box = max(4, line_width * 2)

    for gt in gt_boxes:
        x, y, w, h = gt['bbox']
        x1 = off_x + x * scale_x
        y1 = off_y + y * scale_y
        x2 = off_x + (x + w) * scale_x
        y2 = off_y + (y + h) * scale_y
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        if x2 - x1 < min_box:
            x1 = cx - min_box / 2
            x2 = cx + min_box / 2
        if y2 - y1 < min_box:
            y1 = cy - min_box / 2
            y2 = cy + min_box / 2

        rect = [
            max(0, x1),
            max(0, y1),
            min(canvas.width - 1, x2),
            min(canvas.height - 1, y2),
        ]
        for delta in range(line_width + 1):
            draw.rectangle(
                (rect[0] - delta, rect[1] - delta, rect[2] + delta,
                 rect[3] + delta),
                outline=(0, 0, 0))
        for delta in range(line_width):
            draw.rectangle(
                (rect[0] - delta, rect[1] - delta, rect[2] + delta,
                 rect[3] + delta),
                outline=(255, 230, 0))


def make_panel(image,
               title,
               panel_size,
               font,
               gt_boxes=None,
               gt_line_width=2):
    title_h = 24
    panel = Image.new('RGB', (panel_size, panel_size + title_h),
                      (245, 245, 242))
    panel.paste(
        fit_to_panel(image, panel_size, gt_boxes, gt_line_width),
        (0, title_h))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel_size, title_h), fill=(34, 34, 34))
    draw.text((8, 5), title, fill=(245, 245, 245), font=font)
    return panel


def make_bar_panel(level_stats, panel_size, font):
    title_h = 24
    panel = Image.new('RGB', (panel_size, panel_size + title_h),
                      (245, 245, 242))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel_size, title_h), fill=(34, 34, 34))
    draw.text((8, 5), 'Top gate channels', fill=(245, 245, 245), font=font)

    rows = []
    for level, stats in level_stats:
        top = stats.get('channels', [])[:4]
        rows.append((f'{level} mean top', None))
        for item in top:
            label = f'ch{item["channel"]:02d}'
            value = item['mean']
            ratio = item.get('gt_focus_ratio')
            if ratio is not None:
                label += f' r={ratio:.2f}'
            rows.append((label, value))

    if not rows:
        draw.text((12, title_h + 14), 'No gate stats', fill=(30, 30, 30),
                  font=font)
        return panel

    values = [v for _, v in rows if v is not None]
    max_value = max(max(values), 1e-6) if values else 1.0
    y = title_h + 10
    for label, value in rows:
        if y > panel.height - 16:
            break
        if value is None:
            draw.text((10, y), label, fill=(20, 20, 20), font=font)
            y += 18
            continue
        bar_w = int((panel_size - 92) * value / max_value)
        draw.rectangle((76, y + 2, 76 + bar_w, y + 13),
                       fill=(220, 36, 48))
        draw.text((10, y), label, fill=(20, 20, 20), font=font)
        draw.text((80 + bar_w, y), f'{value:.3f}', fill=(20, 20, 20),
                  font=font)
        y += 18
    return panel


def render_visualization(image_path, gate_items, level_stats, meta, out_path,
                         alpha, panel_size, gt_boxes, gt_line_width):
    original = Image.open(image_path).convert('RGB')
    original_size = original.size
    font = ImageFont.load_default()

    gt_title = f'Original + GT ({len(gt_boxes)})' if gt_boxes else 'Original'
    panels = [
        make_panel(original, gt_title, panel_size, font, gt_boxes,
                   gt_line_width)
    ]

    for item in gate_items:
        gate = item['gate']
        mean_heat = resize_heat_to_original(gate.mean(axis=0), meta,
                                            item['level'], original_size)
        panels.append(
            make_panel(overlay_heatmap(original, mean_heat, alpha),
                       f'{item["level"]} detail gate mean', panel_size, font))

        stats = dict(level_stats).get(item['level'])
        if stats and stats['top_by_mean']:
            top_ch = stats['top_by_mean'][0]
            ch_heat = resize_heat_to_original(gate[top_ch], meta,
                                              item['level'], original_size)
            panels.append(
                make_panel(overlay_heatmap(original, ch_heat, alpha),
                           f'{item["level"]} top ch{top_ch:02d}',
                           panel_size, font))

    panels.append(make_bar_panel(level_stats, panel_size, font))

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


def build_contact_sheet(paths, out_path, thumb_size):
    if not paths:
        return
    cols = 10
    rows = math.ceil(len(paths) / cols)
    sheet = Image.new('RGB', (cols * thumb_size, rows * thumb_size),
                      (250, 250, 250))
    for idx, path in enumerate(paths):
        img = Image.open(path).convert('RGB')
        img.thumbnail((thumb_size, thumb_size), LANCZOS)
        x = (idx % cols) * thumb_size + (thumb_size - img.width) // 2
        y = (idx // cols) * thumb_size + (thumb_size - img.height) // 2
        sheet.paste(img, (x, y))
    sheet.save(out_path, quality=92)


def main():
    args = parse_args()
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
    gt_categories = {}
    gt_ann_file = None
    if not args.no_gt_boxes:
        gt_ann_file = infer_gt_ann_file(cfg)
        if gt_ann_file and gt_ann_file.exists():
            gt_index, gt_categories = load_gt_index(gt_ann_file)
            print(f'Loaded GT boxes from {gt_ann_file}')
        else:
            print('GT boxes disabled: annotation file was not found from '
                  'the test dataloader config.')

    model = init_detector(cfg, str(checkpoint_path), device=device)
    test_pipeline = build_inference_pipeline(model.cfg)
    hooks = register_detail_gate_hooks(model)
    if not hooks:
        raise RuntimeError('No SCPV2DetailGate modules were found.')

    rendered = []
    manifest_items = []
    aggregate = {}
    for idx, image_path in enumerate(sampled, start=1):
        clear_cached_gates(model)
        with torch.no_grad():
            result = inference_detector(model, str(image_path), test_pipeline)

        gate_items = collect_gates(model)
        if not gate_items:
            print(f'[{idx:03d}/{sample_n}] skipped, no gate output: '
                  f'{image_path}')
            continue

        gt_boxes = lookup_gt_boxes(gt_index, image_path)
        original_size = Image.open(image_path).size
        level_stats = []
        for item in gate_items:
            stats = channel_stats(item['gate'], result.metainfo,
                                  item['level'], original_size, gt_boxes,
                                  args.top_k)
            level_stats.append((item['level'], stats))
            agg = aggregate.setdefault(item['level'], {
                'n': 0,
                'gate_mean_sum': 0.0,
                'focus_ratio_sum': 0.0,
                'focus_ratio_n': 0,
                'top_overlap_sum': 0,
                'top_overlap_n': 0,
            })
            agg['n'] += 1
            agg['gate_mean_sum'] += stats['gate_mean']
            if stats['gt_focus_ratio_mean_top_by_mean'] is not None:
                agg['focus_ratio_sum'] += stats[
                    'gt_focus_ratio_mean_top_by_mean']
                agg['focus_ratio_n'] += 1
            if stats['top_overlap_at_k'] is not None:
                agg['top_overlap_sum'] += stats['top_overlap_at_k']
                agg['top_overlap_n'] += 1

        safe_stem = image_path.stem.replace(' ', '_')
        out_path = out_dir / f'{idx:03d}_{safe_stem}_scpv2_gate.png'
        render_visualization(image_path, gate_items, level_stats,
                             result.metainfo, out_path, args.alpha,
                             args.panel_size, gt_boxes, args.gt_line_width)
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
                'name': item['name'],
                'level': item['level'],
                'beta': item['beta'],
                'gate_shape': item['gate_shape'],
                'stats': stats,
            } for item, (_, stats) in zip(gate_items, level_stats)],
        })

        if idx == 1 or idx % 10 == 0 or idx == sample_n:
            print(f'[{idx:03d}/{sample_n}] wrote {out_path}')

    contact_sheet = out_dir / 'contact_sheet.jpg'
    build_contact_sheet(rendered, contact_sheet, args.thumb_size)

    aggregate_summary = {}
    for level, agg in aggregate.items():
        aggregate_summary[level] = {
            'images': agg['n'],
            'gate_mean': agg['gate_mean_sum'] / max(agg['n'], 1),
            'mean_gt_focus_ratio_top_by_mean':
            (agg['focus_ratio_sum'] / agg['focus_ratio_n']
             if agg['focus_ratio_n'] else None),
            'mean_top_overlap_at_k':
            (agg['top_overlap_sum'] / agg['top_overlap_n']
             if agg['top_overlap_n'] else None),
        }

    manifest = {
        'config': str(config_path),
        'checkpoint': str(checkpoint_path),
        'img_dir': str(img_dir),
        'out_dir': str(out_dir),
        'seed': args.seed,
        'sample_n': sample_n,
        'rendered_n': len(rendered),
        'device': device,
        'top_k': args.top_k,
        'alpha': args.alpha,
        'gt_ann_file': str(gt_ann_file) if gt_ann_file else None,
        'gt_category_count': len(gt_categories),
        'gt_boxes_used': not args.no_gt_boxes and bool(gt_index),
        'aggregate_summary': aggregate_summary,
        'contact_sheet': str(contact_sheet),
        'items': manifest_items,
    }
    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    for hook in hooks:
        hook.remove()

    print(f'Wrote {len(rendered)} visualizations to {out_dir}')
    print(f'Contact sheet: {contact_sheet}')
    print(f'Manifest: {out_dir / "manifest.json"}')
    print('How to read: warm yellow/red gate regions mean stronger SCPV2 '
          'detail-gate activation. Check manifest top_by_mean vs '
          'top_by_gt_contrast: high overlap and GT focus ratio > 1 suggest '
          'the gate-selected channels are target-aligned.')


if __name__ == '__main__':
    main()
