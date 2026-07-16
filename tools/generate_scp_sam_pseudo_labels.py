"""Generate SAM box-prompt pseudo labels for SCP diagnostics.

SAM does not predict AITOD categories.  This script uses AITOD GT boxes as
SAM prompts, then writes the matched GT category id (0..7) into the SAM mask.
The resulting ``.npz`` files keep the existing SCP schema:

    label: H x W int32, AITOD class ids 0..7, ignored pixels = 255
    confidence: H x W float16, SAM mask score on object pixels

The purpose is to compare mask quality / noise against SegFormer pseudo
labels, not to replace the final SCPV3 design by GT-only supervision.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

IGNORE_INDEX = 255


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate SAM box-prompt pseudo labels for AITOD.')
    parser.add_argument('--ann-file', required=True, help='AITOD COCO json.')
    parser.add_argument('--img-dir', required=True, help='Image directory.')
    parser.add_argument('--output-dir', required=True, help='Output .npz dir.')
    parser.add_argument(
        '--sam-checkpoint',
        required=True,
        help='Path to SAM/SAM-HQ checkpoint, e.g. sam_vit_h_4b8939.pth.')
    parser.add_argument(
        '--sam-impl',
        choices=('auto', 'sam', 'sam-hq'),
        default='auto',
        help='Which implementation package to use.')
    parser.add_argument(
        '--sam-model-type',
        default='vit_h',
        choices=('vit_h', 'vit_l', 'vit_b'),
        help='SAM backbone type.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--box-expand',
        type=float,
        default=0.10,
        help='Expand each GT box before prompting SAM.')
    parser.add_argument(
        '--mask-clip',
        choices=('box', 'expanded_box', 'none'),
        default='expanded_box',
        help='Clip SAM masks to avoid target leaking into background.')
    parser.add_argument(
        '--min-area-ratio',
        type=float,
        default=0.10,
        help='Reject SAM masks smaller than this ratio of the GT box area.')
    parser.add_argument(
        '--max-area-ratio',
        type=float,
        default=2.00,
        help='Reject SAM masks larger than this ratio of the expanded box area.')
    parser.add_argument(
        '--fallback',
        choices=('box', 'ellipse', 'ignore'),
        default='ellipse',
        help='Fallback mask when SAM output fails sanity checks.')
    parser.add_argument(
        '--overlap-policy',
        choices=('small-first', 'large-first', 'score'),
        default='small-first',
        help='Resolve overlapping object masks.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument(
        '--vis-dir',
        default=None,
        help='Optional directory for quick overlay visualizations.')
    parser.add_argument(
        '--vis-limit',
        type=int,
        default=20,
        help='Max number of visualizations to save.')
    return parser.parse_args()


def load_sam_predictor(args):
    if args.sam_impl in ('sam-hq', 'auto'):
        try:
            from segment_anything_hq import SamPredictor, sam_model_registry
            sam = sam_model_registry[args.sam_model_type](
                checkpoint=args.sam_checkpoint)
            sam.to(device=args.device)
            return SamPredictor(sam)
        except ImportError as hq_exc:
            if args.sam_impl == 'sam-hq':
                raise ImportError(
                    'Cannot import segment_anything_hq. Install it with:\n'
                    '  pip install segment-anything-hq') from hq_exc

    if args.sam_impl in ('sam', 'auto'):
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as sam_exc:
            raise ImportError(
                'Cannot import segment_anything.\n'
                'Install it in the WSL conda env, for example:\n'
                '  source /home/wzc/miniconda3/etc/profile.d/conda.sh\n'
                '  conda activate mmdet5090\n'
                '  pip install git+https://github.com/facebookresearch/'
                'segment-anything.git\n'
                'Then download an official checkpoint and pass '
                '--sam-checkpoint /path/to/sam_vit_h_4b8939.pth.'
            ) from sam_exc

        sam = sam_model_registry[args.sam_model_type](
            checkpoint=args.sam_checkpoint)
        sam.to(device=args.device)
        return SamPredictor(sam)

    raise ValueError(f'Unsupported --sam-impl: {args.sam_impl}')


def load_aitod(ann_file):
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}
    cat_ids = sorted(cat['id'] for cat in data.get('categories', []))
    cat_to_label = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}

    anns_by_img = defaultdict(list)
    for ann in data.get('annotations', []):
        if ann.get('iscrowd', 0):
            continue
        if ann.get('category_id') not in cat_to_label:
            continue
        x, y, w, h = ann.get('bbox', [0, 0, 0, 0])
        if w <= 0 or h <= 0:
            continue
        anns_by_img[ann['image_id']].append(ann)
    return images, anns_by_img, cat_to_label


def expand_box(bbox, img_w, img_h, ratio):
    x, y, w, h = [float(v) for v in bbox]
    cx, cy = x + 0.5 * w, y + 0.5 * h
    ew, eh = w * (1.0 + ratio), h * (1.0 + ratio)
    x1 = max(0, int(np.floor(cx - 0.5 * ew)))
    y1 = max(0, int(np.floor(cy - 0.5 * eh)))
    x2 = min(img_w, int(np.ceil(cx + 0.5 * ew)))
    y2 = min(img_h, int(np.ceil(cy + 0.5 * eh)))
    return x1, y1, x2, y2


def box_mask(bbox, img_h, img_w, expand=0.0):
    x1, y1, x2, y2 = expand_box(bbox, img_w, img_h, expand)
    mask = np.zeros((img_h, img_w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def ellipse_mask(bbox, img_h, img_w):
    x1, y1, x2, y2 = expand_box(bbox, img_w, img_h, 0.0)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cx = int(round((x1 + x2 - 1) * 0.5))
    cy = int(round((y1 + y2 - 1) * 0.5))
    axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1, thickness=-1)
    return mask.astype(bool)


def select_sam_mask(predictor, prompt_box, ann, img_h, img_w, args):
    masks, scores, _ = predictor.predict(
        box=np.asarray(prompt_box, dtype=np.float32),
        multimask_output=True)

    gt_area = max(float(ann.get('area', ann['bbox'][2] * ann['bbox'][3])), 1.0)
    expanded_area = max(
        float((prompt_box[2] - prompt_box[0]) * (prompt_box[3] - prompt_box[1])),
        1.0)
    order = np.argsort(scores)[::-1]
    for idx in order:
        mask = masks[idx].astype(bool)
        area = float(mask.sum())
        if area < args.min_area_ratio * gt_area:
            continue
        if area > args.max_area_ratio * expanded_area:
            continue
        return mask, float(scores[idx]), False

    if args.fallback == 'box':
        return box_mask(ann['bbox'], img_h, img_w), 0.50, True
    if args.fallback == 'ellipse':
        return ellipse_mask(ann['bbox'], img_h, img_w), 0.50, True
    return np.zeros((img_h, img_w), dtype=bool), 0.0, True


def paint_instances(label, confidence, instances, overlap_policy):
    if overlap_policy == 'small-first':
        instances = sorted(instances, key=lambda item: item['area'])
    elif overlap_policy == 'large-first':
        instances = sorted(
            instances, key=lambda item: item['area'], reverse=True)
    else:
        instances = sorted(
            instances, key=lambda item: item['score'], reverse=True)

    occupied = np.zeros(label.shape, dtype=bool)
    for inst in instances:
        mask = inst['mask'] & ~occupied
        if not mask.any():
            continue
        label[mask] = inst['label']
        confidence[mask] = inst['score']
        occupied[mask] = True


def save_visualization(path, image_rgb, label, ignore_index=IGNORE_INDEX):
    palette = np.asarray([
        [220, 20, 60],
        [119, 11, 32],
        [0, 0, 142],
        [0, 0, 230],
        [106, 0, 228],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 70],
    ], dtype=np.uint8)
    overlay = image_rgb.copy()
    valid = label != ignore_index
    if valid.any():
        colors = palette[np.clip(label[valid], 0, len(palette) - 1)]
        overlay[valid] = (0.55 * overlay[valid] + 0.45 * colors).astype(
            np.uint8)
    Image.fromarray(overlay).save(path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

    images, anns_by_img, cat_to_label = load_aitod(args.ann_file)
    image_items = list(images.items())
    if args.max_images is not None:
        image_items = image_items[:args.max_images]

    predictor = load_sam_predictor(args)

    generated = 0
    skipped = 0
    fallback_count = 0
    object_count = 0

    for img_id, img_info in tqdm(image_items, desc='Generating SAM labels'):
        stem = Path(img_info['file_name']).stem
        output_path = Path(args.output_dir) / f'{stem}.npz'
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        img_path = Path(args.img_dir) / img_info['file_name']
        image_rgb = np.asarray(Image.open(img_path).convert('RGB'))
        img_h, img_w = image_rgb.shape[:2]
        predictor.set_image(image_rgb)

        label = np.full((img_h, img_w), IGNORE_INDEX, dtype=np.int32)
        confidence = np.zeros((img_h, img_w), dtype=np.float16)
        instances = []

        for ann in anns_by_img.get(img_id, []):
            prompt_box = expand_box(ann['bbox'], img_w, img_h, args.box_expand)
            mask, score, is_fallback = select_sam_mask(
                predictor, prompt_box, ann, img_h, img_w, args)
            if args.mask_clip == 'box':
                mask = mask & box_mask(ann['bbox'], img_h, img_w)
            elif args.mask_clip == 'expanded_box':
                mask = mask & box_mask(
                    ann['bbox'], img_h, img_w, expand=args.box_expand)

            if is_fallback:
                fallback_count += 1
            object_count += 1
            instances.append(
                dict(
                    mask=mask,
                    label=cat_to_label[ann['category_id']],
                    score=score,
                    area=float(ann.get('area', mask.sum()))))

        paint_instances(label, confidence, instances, args.overlap_policy)
        np.savez_compressed(
            output_path,
            label=label.astype(np.int32, copy=False),
            confidence=confidence.astype(np.float16, copy=False))
        generated += 1

        if args.vis_dir and generated <= args.vis_limit:
            save_visualization(
                Path(args.vis_dir) / f'{stem}.jpg', image_rgb, label)

    print(f'Done. Generated {generated}, skipped {skipped}.')
    print(f'Objects processed: {object_count}. Fallback masks: {fallback_count}.')
    if object_count:
        print(f'Fallback ratio: {fallback_count / object_count:.4f}.')


if __name__ == '__main__':
    main()
