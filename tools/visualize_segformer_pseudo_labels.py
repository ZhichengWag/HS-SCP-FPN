"""Visualize SegFormer coarse pseudo labels for SCP.

Randomly samples per-image ``.npz`` pseudo-label maps, overlays them on the
matching AI-TOD image, and writes individual visualizations plus a contact
sheet and manifest.
"""

import argparse
import colorsys
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from generate_scp_pseudo_labels import map_ade20k_to_coarse


COARSE_CLASS_NAMES = (
    'water',
    'road/runway',
    'building',
    'airport',
    'bridge',
    'farmland',
    'vegetation',
    'other',
)

COARSE_PALETTE = np.array(
    [
        (31, 119, 180),   # water
        (127, 127, 127),  # road/runway
        (214, 39, 40),    # building
        (255, 127, 14),   # airport
        (148, 103, 189),  # bridge
        (188, 189, 34),   # farmland
        (44, 160, 44),    # vegetation
        (140, 86, 75),    # other
    ],
    dtype=np.uint8,
)

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
NEAREST = getattr(getattr(Image, 'Resampling', Image), 'NEAREST')
LANCZOS = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize 8-class SegFormer SCP pseudo labels.')
    parser.add_argument(
        '--img-dir',
        default=r'E:\AI-TOD\trainval\images',
        help='Directory containing original images.')
    parser.add_argument(
        '--pseudo-dir',
        default=r'E:\AI-TOD\pseudo_labels\trainval',
        help='Directory containing per-image .npz pseudo labels.')
    parser.add_argument(
        '--out-dir',
        default=r'E:\mmdet5090\visualizations\segformer_pseudo_8class_100',
        help='Directory to write visualizations.')
    parser.add_argument('--sample-n', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260603)
    parser.add_argument('--alpha', type=float, default=0.45)
    parser.add_argument('--key', default='label', help='Preferred npz key.')
    parser.add_argument(
        '--input-label-space',
        choices=('coarse8', 'ade20k'),
        default='coarse8',
        help='Map raw ADE20K labels to the 8-class display palette if needed.')
    parser.add_argument(
        '--display-label-space',
        choices=('coarse8', 'ade20k'),
        default='coarse8',
        help='Display raw ADE20K classes or the mapped 8-class palette.')
    parser.add_argument(
        '--thumb-size',
        type=int,
        default=220,
        help='Thumbnail size for the contact sheet.')
    return parser.parse_args()


def load_label(path, key, input_label_space, display_label_space):
    with np.load(path) as data:
        if key in data:
            label = data[key]
        elif 'arr_0' in data:
            label = data['arr_0']
        elif data.files:
            label = data[data.files[0]]
        else:
            raise KeyError(f'No arrays found in {path}')

    if label.ndim == 3 and label.shape[0] == 1:
        label = label.squeeze(0)
    if label.ndim != 2:
        raise ValueError(f'Expected 2D label map, got {label.shape}: {path}')
    if input_label_space == 'coarse8' and display_label_space == 'ade20k':
        raise ValueError('Cannot recover ADE20K labels from coarse8 labels.')
    if input_label_space == 'ade20k' and display_label_space == 'coarse8':
        label = map_ade20k_to_coarse(label)
    return label.astype(np.uint8, copy=False)


def find_image(img_dir, stem):
    for ext in IMAGE_EXTS:
        path = img_dir / f'{stem}{ext}'
        if path.exists():
            return path
    return None


def make_ade20k_palette(num_classes=150):
    colors = []
    for idx in range(num_classes):
        rgb = colorsys.hsv_to_rgb((idx * 0.61803398875) % 1.0, 0.72, 0.95)
        colors.append(tuple(round(channel * 255) for channel in rgb))
    return np.asarray(colors, dtype=np.uint8)


def get_display_meta(label_space):
    if label_space == 'coarse8':
        return COARSE_CLASS_NAMES, COARSE_PALETTE

    from transformers import SegformerConfig
    config = SegformerConfig.from_pretrained(
        'nvidia/segformer-b5-finetuned-ade-640-640',
        local_files_only=True)
    class_names = tuple(
        config.id2label.get(idx, str(idx)) for idx in range(150))
    return class_names, make_ade20k_palette()


def draw_legend(draw, x, y, font, class_names, palette, class_ids):
    cursor_x = x
    cursor_y = y
    row_height = 20
    for position, class_id in enumerate(class_ids):
        if position == 4:
            cursor_x = x
            cursor_y += row_height
        color = tuple(int(v) for v in palette[class_id])
        draw.rectangle([cursor_x, cursor_y + 3, cursor_x + 12, cursor_y + 15],
                       fill=color)
        name = class_names[class_id]
        draw.text((cursor_x + 16, cursor_y), f'{class_id}: {name}',
                  fill=(20, 20, 20), font=font)
        cursor_x += 190


def render_overlay(image_path, label_path, key, alpha, input_label_space,
                   display_label_space, class_names, palette):
    image = Image.open(image_path).convert('RGB')
    label = load_label(label_path, key, input_label_space,
                       display_label_space)

    if label.shape != (image.height, image.width):
        label_img = Image.fromarray(label, mode='L')
        label_img = label_img.resize((image.width, image.height),
                                     NEAREST)
        label = np.asarray(label_img, dtype=np.uint8)

    clamped = np.where(label < len(class_names), label, 0)
    color = palette[clamped]
    valid = label < len(class_names)

    base = np.asarray(image, dtype=np.float32)
    blended = base.copy()
    blended[valid] = base[valid] * (1.0 - alpha) + color[valid] * alpha
    overlay = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    legend_h = 78
    canvas = Image.new('RGB', (overlay.width, overlay.height + legend_h),
                       (245, 245, 242))
    canvas.paste(overlay, (0, 0))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    title = f'{image_path.name} | pseudo: {label_path.name}'
    draw.rectangle([0, overlay.height, canvas.width, canvas.height],
                   fill=(245, 245, 242))
    draw.text((10, overlay.height + 8), title, fill=(20, 20, 20), font=font)
    counts = np.bincount(label[valid], minlength=len(class_names))
    top_ids = np.argsort(counts)[::-1]
    top_ids = [int(idx) for idx in top_ids if counts[idx] > 0][:8]
    draw_legend(draw, 10, overlay.height + 30, font, class_names, palette,
                top_ids)

    hist = {
        f'{i}: {class_names[i]}': int(counts[i])
        for i in range(len(class_names)) if counts[i] > 0
    }
    ignored = int(np.sum(label >= len(class_names)))
    if ignored:
        hist['ignored_or_unknown'] = ignored

    return canvas, hist


def build_contact_sheet(paths, out_path, thumb_size):
    if not paths:
        return
    cols = 10
    rows = math.ceil(len(paths) / cols)
    sheet = Image.new('RGB', (cols * thumb_size, rows * thumb_size),
                      (250, 250, 250))
    for i, path in enumerate(paths):
        img = Image.open(path).convert('RGB')
        img.thumbnail((thumb_size, thumb_size), LANCZOS)
        x = (i % cols) * thumb_size + (thumb_size - img.width) // 2
        y = (i // cols) * thumb_size + (thumb_size - img.height) // 2
        sheet.paste(img, (x, y))
    sheet.save(out_path, quality=92)


def main():
    args = parse_args()
    img_dir = Path(args.img_dir)
    pseudo_dir = Path(args.pseudo_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for pseudo_path in sorted(pseudo_dir.glob('*.npz')):
        image_path = find_image(img_dir, pseudo_path.stem)
        if image_path is not None:
            candidates.append((image_path, pseudo_path))

    if not candidates:
        raise RuntimeError(
            f'No matching image/.npz pairs found in {img_dir} and {pseudo_dir}')

    rng = random.Random(args.seed)
    sample_n = min(args.sample_n, len(candidates))
    sampled = rng.sample(candidates, sample_n)
    class_names, palette = get_display_meta(args.display_label_space)

    items = []
    rendered_paths = []
    total_hist = {f'{i}: {name}': 0 for i, name in enumerate(class_names)}
    for idx, (image_path, pseudo_path) in enumerate(sampled, start=1):
        canvas, hist = render_overlay(image_path, pseudo_path, args.key,
                                      args.alpha, args.input_label_space,
                                      args.display_label_space, class_names,
                                      palette)
        out_name = f'{idx:03d}_{image_path.stem}.png'
        out_path = out_dir / out_name
        canvas.save(out_path)
        rendered_paths.append(out_path)

        for name in total_hist:
            total_hist[name] += hist.get(name, 0)
        item = {
            'index': idx,
            'image': str(image_path),
            'pseudo_label': str(pseudo_path),
            'output': str(out_path),
            'pixel_histogram': hist,
        }
        items.append(item)

    contact_sheet = out_dir / 'contact_sheet.jpg'
    build_contact_sheet(rendered_paths, contact_sheet, args.thumb_size)

    manifest = {
        'seed': args.seed,
        'sample_n': sample_n,
        'img_dir': str(img_dir),
        'pseudo_dir': str(pseudo_dir),
        'out_dir': str(out_dir),
        'input_label_space': args.input_label_space,
        'display_label_space': args.display_label_space,
        'class_names': list(class_names),
        'palette_rgb': palette.tolist(),
        'total_pixel_histogram': total_hist,
        'contact_sheet': str(contact_sheet),
        'items': items,
    }
    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f'Wrote {sample_n} visualizations to {out_dir}')
    print(f'Contact sheet: {contact_sheet}')
    print(f'Manifest: {out_dir / "manifest.json"}')


if __name__ == '__main__':
    main()
