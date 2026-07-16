"""Evaluate SCP pseudo-label quality against AITOD training annotations.

This diagnostic is designed for comparing full-image SegFormer labels with
GT-box-prompted SAM labels.  It reports:

  - any_fg precision/recall/IoU: valid pseudo pixels vs GT object regions
  - class-aware precision/recall/IoU: valid pixels that match AITOD gt class
  - valid pixel ratio: how much of each image is supervised

For SegFormer raw ADE20K labels, class-aware metrics are not meaningful unless
the labels were mapped to AITOD 0..7.  Any-foreground metrics still show how
much background noise would enter distillation.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IGNORE_INDEX = 255


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate SCP pseudo-label quality on AITOD.')
    parser.add_argument('--ann-file', required=True, help='AITOD COCO json.')
    parser.add_argument('--pseudo-dir', required=True, help='Pseudo .npz dir.')
    parser.add_argument('--output-csv', default=None)
    parser.add_argument('--key', default='label')
    parser.add_argument('--confidence-key', default='confidence')
    parser.add_argument('--confidence-thr', type=float, default=None)
    parser.add_argument('--ignore-index', type=int, default=IGNORE_INDEX)
    parser.add_argument(
        '--gt-shape',
        choices=('segmentation', 'box'),
        default='segmentation',
        help='Use AITOD segmentation polygons or boxes as GT masks.')
    parser.add_argument(
        '--label-space',
        choices=('aitod8', 'any'),
        default='aitod8',
        help='aitod8 enables class-aware metrics for labels 0..7.')
    parser.add_argument('--max-images', type=int, default=None)
    return parser.parse_args()


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


def rasterize_gt(img_info, anns, cat_to_label, gt_shape):
    img_h = int(img_info['height'])
    img_w = int(img_info['width'])
    gt_any = np.zeros((img_h, img_w), dtype=bool)
    gt_class = np.full((img_h, img_w), IGNORE_INDEX, dtype=np.int32)

    # Large first lets smaller later annotations overwrite local class labels.
    anns = sorted(anns, key=lambda ann: ann.get('area', 0), reverse=True)
    for ann in anns:
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if gt_shape == 'segmentation':
            seg = ann.get('segmentation', None)
            polys = []
            if isinstance(seg, list):
                for poly in seg:
                    arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                    if arr.shape[0] >= 3:
                        polys.append(np.round(arr).astype(np.int32))
            if polys:
                cv2.fillPoly(mask, polys, 1)
            else:
                fill_box(mask, ann['bbox'])
        else:
            fill_box(mask, ann['bbox'])

        obj = mask.astype(bool)
        gt_any |= obj
        gt_class[obj] = cat_to_label[ann['category_id']]
    return gt_any, gt_class


def fill_box(mask, bbox):
    x, y, w, h = bbox
    img_h, img_w = mask.shape[:2]
    x1 = max(0, int(np.floor(x)))
    y1 = max(0, int(np.floor(y)))
    x2 = min(img_w, int(np.ceil(x + w)))
    y2 = min(img_h, int(np.ceil(y + h)))
    mask[y1:y2, x1:x2] = 1


def load_pseudo(path, args):
    with np.load(path) as data:
        if args.key in data:
            label = data[args.key]
        elif 'arr_0' in data:
            label = data['arr_0']
        else:
            label = data[data.files[0]]

        confidence = None
        if args.confidence_key in data:
            confidence = data[args.confidence_key]

    if label.ndim == 3 and label.shape[0] == 1:
        label = label.squeeze(0)
    if confidence is not None and confidence.ndim == 3 and confidence.shape[0] == 1:
        confidence = confidence.squeeze(0)
    return label.astype(np.int32, copy=False), confidence


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def main():
    args = parse_args()
    images, anns_by_img, cat_to_label = load_aitod(args.ann_file)
    image_items = list(images.items())
    if args.max_images is not None:
        image_items = image_items[:args.max_images]

    totals = defaultdict(float)
    rows = []
    missing = 0

    for img_id, img_info in tqdm(image_items, desc='Evaluating labels'):
        pseudo_path = Path(args.pseudo_dir) / f"{Path(img_info['file_name']).stem}.npz"
        if not pseudo_path.exists():
            missing += 1
            continue

        label, confidence = load_pseudo(pseudo_path, args)
        gt_any, gt_class = rasterize_gt(
            img_info, anns_by_img.get(img_id, []), cat_to_label, args.gt_shape)

        if label.shape != gt_any.shape:
            raise ValueError(
                f'Pseudo shape mismatch for {pseudo_path}: '
                f'{label.shape} vs {gt_any.shape}')

        pred_valid = label != args.ignore_index
        if confidence is not None and args.confidence_thr is not None:
            pred_valid &= confidence >= args.confidence_thr

        any_inter = np.logical_and(pred_valid, gt_any).sum()
        any_union = np.logical_or(pred_valid, gt_any).sum()
        pred_valid_sum = pred_valid.sum()
        gt_sum = gt_any.sum()

        class_valid = pred_valid & (label >= 0) & (label < 8)
        if args.label_space == 'aitod8':
            class_match = class_valid & (label == gt_class)
        else:
            class_match = np.zeros_like(pred_valid, dtype=bool)
        class_inter = class_match.sum()
        class_union = np.logical_or(class_valid, gt_class != args.ignore_index).sum()

        row = dict(
            file_name=img_info['file_name'],
            valid_pixels=int(pred_valid_sum),
            gt_pixels=int(gt_sum),
            valid_ratio=safe_div(pred_valid_sum, label.size),
            any_precision=safe_div(any_inter, pred_valid_sum),
            any_recall=safe_div(any_inter, gt_sum),
            any_iou=safe_div(any_inter, any_union),
            class_precision=safe_div(class_inter, class_valid.sum()),
            class_recall=safe_div(class_inter, gt_sum),
            class_iou=safe_div(class_inter, class_union),
        )
        rows.append(row)

        totals['valid_pixels'] += pred_valid_sum
        totals['gt_pixels'] += gt_sum
        totals['pixels'] += label.size
        totals['any_inter'] += any_inter
        totals['any_union'] += any_union
        totals['class_valid'] += class_valid.sum()
        totals['class_inter'] += class_inter
        totals['class_union'] += class_union

    summary = dict(
        images=len(rows),
        missing=missing,
        valid_ratio=safe_div(totals['valid_pixels'], totals['pixels']),
        any_precision=safe_div(totals['any_inter'], totals['valid_pixels']),
        any_recall=safe_div(totals['any_inter'], totals['gt_pixels']),
        any_iou=safe_div(totals['any_inter'], totals['any_union']),
        class_precision=safe_div(totals['class_inter'], totals['class_valid']),
        class_recall=safe_div(totals['class_inter'], totals['gt_pixels']),
        class_iou=safe_div(totals['class_inter'], totals['class_union']),
    )

    print('Summary')
    for key, value in summary.items():
        if isinstance(value, float):
            print(f'  {key}: {value:.6f}')
        else:
            print(f'  {key}: {value}')

    if args.output_csv:
        fieldnames = [
            'file_name', 'valid_pixels', 'gt_pixels', 'valid_ratio',
            'any_precision', 'any_recall', 'any_iou', 'class_precision',
            'class_recall', 'class_iou'
        ]
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
