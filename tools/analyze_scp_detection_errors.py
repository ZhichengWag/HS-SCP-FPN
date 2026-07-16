"""Diagnose SCP detector errors as localization or classification failures.

The script runs MMDetection inference with a config/checkpoint, matches
predictions to COCO-style GT boxes, and writes a compact error report.
It is meant for answering: are most errors caused by inaccurate boxes, or by
wrong classes after the object has already been localized?
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MMDET_ROOT = WORKSPACE_ROOT / 'mmdetection'
for path in (str(WORKSPACE_ROOT), str(MMDET_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


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
        description='Analyze whether detector errors are localization or '
        'classification failures.')
    parser.add_argument(
        '--config',
        default=str(WORKSPACE_ROOT / 'config_hsfpn' /
                    'cascade_rcnn_r50_aitod_scp.py'),
        help='MMDetection config path.')
    parser.add_argument(
        '--checkpoint',
        default=str(WORKSPACE_ROOT / 'work_dirs' /
                    'cascade_rcnn_r50_aitod_scp_b1_0.005' /
                    'best_coco_bbox_mAP_epoch_12.pth'),
        help='Checkpoint path.')
    parser.add_argument(
        '--ann-file',
        default=None,
        help='COCO annotation file. Defaults to config test_dataloader.')
    parser.add_argument(
        '--img-dir',
        default=None,
        help='Image directory. Defaults to config test_dataloader.')
    parser.add_argument(
        '--out-dir',
        default=str(WORKSPACE_ROOT / 'visualizations' /
                    'scp_error_analysis'),
        help='Directory for JSON/CSV reports.')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--score-thr', type=float, default=0.05)
    parser.add_argument(
        '--tp-iou',
        type=float,
        default=0.50,
        help='IoU threshold for a correct localization.')
    parser.add_argument(
        '--loc-iou',
        type=float,
        default=0.10,
        help='Loose IoU threshold for saying a prediction touched an object.')
    parser.add_argument(
        '--max-per-img',
        type=int,
        default=300,
        help='Keep at most this many predictions per image after thresholding.')
    parser.add_argument(
        '--sample-n',
        type=int,
        default=0,
        help='Analyze a random subset. 0 means all images in ann_file.')
    parser.add_argument('--seed', type=int, default=20260607)
    parser.add_argument(
        '--top-error-examples',
        type=int,
        default=100,
        help='Maximum per-image error examples saved in detail JSON.')
    return parser.parse_args()


def resolve_device(device):
    if device == 'auto':
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device


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


def resolve_dataset_paths(cfg, ann_file=None, img_dir=None):
    dataset_cfg = unwrap_dataset_cfg(cfg.test_dataloader.get('dataset', {}))
    data_root = normalize_path(dataset_cfg.get('data_root',
                                               cfg.get('data_root', '')))

    if ann_file is None:
        ann_file = dataset_cfg.get('ann_file', None)
        if ann_file is None:
            raise ValueError('Cannot infer ann_file from test_dataloader.')
    ann_path = normalize_path(ann_file)
    if not ann_path.is_absolute():
        ann_path = data_root / ann_path

    if img_dir is None:
        data_prefix = dataset_cfg.get('data_prefix', {})
        img_prefix = data_prefix.get('img', '') if isinstance(
            data_prefix, dict) else str(data_prefix)
        img_dir = img_prefix
    img_path = normalize_path(img_dir)
    if not img_path.is_absolute():
        img_path = data_root / img_path
    return ann_path, img_path, dataset_cfg


def load_coco_gt(ann_file, img_dir, classes):
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    categories = data.get('categories', [])
    cat_name = {cat['id']: cat.get('name', str(cat['id'])) for cat in categories}
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    sorted_cat_ids = [cat['id'] for cat in sorted(categories,
                                                  key=lambda x: x['id'])]
    sorted_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(sorted_cat_ids)}
    cat_id_to_label = {}
    for cat in categories:
        name = cat.get('name', str(cat['id']))
        cat_id_to_label[cat['id']] = class_to_idx.get(
            name, sorted_id_to_idx.get(cat['id'], 0))

    images = []
    by_id = {}
    for img in data.get('images', []):
        path = img_dir / img.get('file_name', '')
        item = {
            'id': img['id'],
            'file_name': img.get('file_name', ''),
            'path': path,
            'width': img.get('width'),
            'height': img.get('height'),
        }
        images.append(item)
        by_id[img['id']] = item

    gt_by_image = defaultdict(list)
    for ann in data.get('annotations', []):
        if ann.get('ignore', 0) or ann.get('iscrowd', 0):
            continue
        bbox = ann.get('bbox')
        if bbox is None or len(bbox) != 4:
            continue
        x, y, w, h = [float(v) for v in bbox]
        if w <= 0 or h <= 0:
            continue
        cat_id = ann.get('category_id')
        label = cat_id_to_label.get(cat_id)
        if label is None:
            continue
        gt_by_image[ann['image_id']].append({
            'ann_id': ann.get('id'),
            'bbox': [x, y, x + w, y + h],
            'label': int(label),
            'category_id': cat_id,
            'category_name': cat_name.get(cat_id, str(cat_id)),
            'area': float(ann.get('area', w * h)),
        })
    return images, gt_by_image, cat_id_to_label


def xyxy_iou(boxes1, boxes2):
    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], 0, None) * np.clip(
        boxes1[:, 3] - boxes1[:, 1], 0, None)
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], 0, None) * np.clip(
        boxes2[:, 3] - boxes2[:, 1], 0, None)
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, 1e-12, None)


def result_to_predictions(result, score_thr, max_per_img):
    pred = result.pred_instances
    bboxes = pred.bboxes.detach().cpu().numpy()
    labels = pred.labels.detach().cpu().numpy().astype(np.int64)
    scores = pred.scores.detach().cpu().numpy()
    keep = scores >= score_thr
    bboxes = bboxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    order = np.argsort(scores)[::-1]
    if max_per_img > 0:
        order = order[:max_per_img]
    return [{
        'bbox': bboxes[i].astype(float).tolist(),
        'label': int(labels[i]),
        'score': float(scores[i]),
    } for i in order]


def greedy_match_tps(preds, gts, ious, tp_iou):
    matched_gt = set()
    matched_pred = set()
    tp_pairs = []
    for pred_idx, pred in enumerate(preds):
        best_gt = None
        best_iou = -1.0
        for gt_idx, gt in enumerate(gts):
            if gt_idx in matched_gt or pred['label'] != gt['label']:
                continue
            iou = float(ious[pred_idx, gt_idx])
            if iou >= tp_iou and iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_gt is not None:
            matched_pred.add(pred_idx)
            matched_gt.add(best_gt)
            tp_pairs.append((pred_idx, best_gt, best_iou))
    return tp_pairs, matched_pred, matched_gt


def class_name(label, classes):
    if 0 <= int(label) < len(classes):
        return classes[int(label)]
    return f'class_{label}'


def diagnose_image(image, preds, gts, classes, tp_iou, loc_iou):
    pred_boxes = [p['bbox'] for p in preds]
    gt_boxes = [g['bbox'] for g in gts]
    ious = xyxy_iou(pred_boxes, gt_boxes)
    tp_pairs, matched_pred, matched_gt = greedy_match_tps(
        preds, gts, ious, tp_iou)

    gt_errors = []
    gt_error_counts = Counter()
    for gt_idx, gt in enumerate(gts):
        if gt_idx in matched_gt:
            gt_error_counts['true_positive'] += 1
            continue
        if not preds:
            best_any_iou = 0.0
            best_any_idx = None
            best_same_iou = 0.0
            best_same_idx = None
        else:
            any_ious = ious[:, gt_idx]
            best_any_idx = int(np.argmax(any_ious))
            best_any_iou = float(any_ious[best_any_idx])
            same_indices = [
                i for i, pred in enumerate(preds)
                if pred['label'] == gt['label']
            ]
            if same_indices:
                same_ious = any_ious[same_indices]
                local = int(np.argmax(same_ious))
                best_same_idx = same_indices[local]
                best_same_iou = float(same_ious[local])
            else:
                best_same_idx = None
                best_same_iou = 0.0

        if best_any_iou >= tp_iou:
            reason = 'classification_error'
        elif best_same_iou >= loc_iou:
            reason = 'localization_error'
        elif best_any_iou >= loc_iou:
            reason = 'classification_and_localization_error'
        else:
            reason = 'missed_background'
        gt_error_counts[reason] += 1

        best_pred = preds[best_any_idx] if best_any_idx is not None else None
        gt_errors.append({
            'reason':
            reason,
            'gt_label':
            gt['label'],
            'gt_class':
            class_name(gt['label'], classes),
            'best_iou_any':
            best_any_iou,
            'best_iou_same_class':
            best_same_iou,
            'best_pred_label':
            best_pred['label'] if best_pred else None,
            'best_pred_class':
            class_name(best_pred['label'], classes) if best_pred else None,
            'best_pred_score':
            best_pred['score'] if best_pred else None,
        })

    fp_errors = []
    fp_error_counts = Counter()
    for pred_idx, pred in enumerate(preds):
        if pred_idx in matched_pred:
            continue
        if not gts:
            max_iou = 0.0
            best_gt_idx = None
            max_same_iou = 0.0
        else:
            pred_ious = ious[pred_idx]
            best_gt_idx = int(np.argmax(pred_ious))
            max_iou = float(pred_ious[best_gt_idx])
            same_indices = [
                i for i, gt in enumerate(gts) if gt['label'] == pred['label']
            ]
            max_same_iou = (float(pred_ious[same_indices].max())
                            if same_indices else 0.0)

        best_gt = gts[best_gt_idx] if best_gt_idx is not None else None
        if best_gt is not None and max_same_iou >= tp_iou:
            reason = 'duplicate_fp'
        elif best_gt is not None and max_iou >= tp_iou:
            reason = 'classification_fp'
        elif max_same_iou >= loc_iou:
            reason = 'localization_fp'
        elif max_iou >= loc_iou:
            reason = 'classification_and_localization_fp'
        else:
            reason = 'background_fp'
        fp_error_counts[reason] += 1
        fp_errors.append({
            'reason':
            reason,
            'pred_label':
            pred['label'],
            'pred_class':
            class_name(pred['label'], classes),
            'score':
            pred['score'],
            'best_iou_any':
            max_iou,
            'best_iou_same_class':
            max_same_iou,
            'best_gt_label':
            best_gt['label'] if best_gt else None,
            'best_gt_class':
            class_name(best_gt['label'], classes) if best_gt else None,
        })

    return {
        'image_id': image['id'],
        'file_name': image['file_name'],
        'gt_count': len(gts),
        'pred_count': len(preds),
        'tp_count': len(tp_pairs),
        'gt_error_counts': dict(gt_error_counts),
        'fp_error_counts': dict(fp_error_counts),
        'gt_errors': gt_errors,
        'fp_errors': fp_errors,
    }


def update_class_stats(class_stats, gts, image_report):
    gt_errors = image_report['gt_errors']
    for gt in gts:
        stats = class_stats[gt['label']]
        stats['gt_total'] += 1
    for item in gt_errors:
        stats = class_stats[item['gt_label']]
        stats[item['reason']] += 1

    tp = image_report['gt_error_counts'].get('true_positive', 0)
    # True positives are class-specific through matched GT, but the compact
    # image report does not store all pairs. Recover by subtracting errors per
    # class after the full loop in finalize_class_stats.
    return tp


def finalize_class_stats(class_stats):
    error_keys = [
        'classification_error',
        'localization_error',
        'classification_and_localization_error',
        'missed_background',
    ]
    out = {}
    for label, stats in sorted(class_stats.items()):
        total = stats.get('gt_total', 0)
        errors = sum(stats.get(k, 0) for k in error_keys)
        true_positive = max(total - errors, 0)
        out[str(label)] = {
            'gt_total': total,
            'true_positive': true_positive,
            **{k: stats.get(k, 0) for k in error_keys},
            'class_accuracy_proxy': true_positive / max(total, 1),
        }
    return out


def dominant_error(summary):
    loc = summary['gt_error_counts'].get('localization_error', 0)
    cls = summary['gt_error_counts'].get('classification_error', 0)
    both = summary['gt_error_counts'].get(
        'classification_and_localization_error', 0)
    miss = summary['gt_error_counts'].get('missed_background', 0)
    candidates = {
        'localization_error': loc,
        'classification_error': cls,
        'classification_and_localization_error': both,
        'missed_background': miss,
    }
    top_key, top_value = max(candidates.items(), key=lambda x: x[1])
    total_errors = sum(candidates.values())
    return {
        'dominant_error': top_key,
        'dominant_error_count': top_value,
        'dominant_error_ratio_among_gt_errors':
        top_value / max(total_errors, 1),
        'interpretation':
        {
            'classification_error':
            'GT has a prediction with good IoU but the predicted class is wrong.',
            'localization_error':
            'GT has a same-class prediction nearby, but IoU is below the TP threshold.',
            'classification_and_localization_error':
            'GT is touched only by wrong-class and low-IoU predictions.',
            'missed_background':
            'No prediction overlaps this GT even under the loose IoU threshold.',
        }[top_key],
    }


def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    config_path = normalize_path(args.config)
    checkpoint_path = normalize_path(args.checkpoint)
    out_dir = normalize_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(config_path))
    ann_file, img_dir, dataset_cfg = resolve_dataset_paths(
        cfg, args.ann_file, args.img_dir)
    if not ann_file.exists():
        raise FileNotFoundError(f'Annotation file not found: {ann_file}')
    if not img_dir.exists():
        raise FileNotFoundError(f'Image dir not found: {img_dir}')

    classes = tuple(dataset_cfg.get('metainfo', {}).get('classes', ()))
    if not classes:
        classes = tuple(cfg.get('aitod_metainfo', {}).get('classes', ()))
    images, gt_by_image, _ = load_coco_gt(ann_file, img_dir, classes)
    if args.sample_n and args.sample_n > 0:
        rng = random.Random(args.seed)
        images = rng.sample(images, min(args.sample_n, len(images)))

    from mmdet.apis import inference_detector, init_detector

    device = resolve_device(args.device)
    model = init_detector(cfg, str(checkpoint_path), device=device)
    if not classes:
        classes = tuple(model.dataset_meta.get('classes', ()))
    test_pipeline = build_inference_pipeline(model.cfg)

    summary = {
        'config': str(config_path),
        'checkpoint': str(checkpoint_path),
        'ann_file': str(ann_file),
        'img_dir': str(img_dir),
        'device': device,
        'score_thr': args.score_thr,
        'tp_iou': args.tp_iou,
        'loc_iou': args.loc_iou,
        'max_per_img': args.max_per_img,
        'images': 0,
        'gt_total': 0,
        'pred_total': 0,
        'tp_total': 0,
        'gt_error_counts': Counter(),
        'fp_error_counts': Counter(),
    }
    per_image = []
    class_stats = defaultdict(Counter)

    for idx, image in enumerate(images, start=1):
        if not image['path'].exists():
            print(f'[{idx:05d}/{len(images)}] missing image: {image["path"]}')
            continue
        result = inference_detector(model, str(image['path']), test_pipeline)
        preds = result_to_predictions(result, args.score_thr,
                                      args.max_per_img)
        gts = gt_by_image.get(image['id'], [])
        report = diagnose_image(image, preds, gts, classes, args.tp_iou,
                                args.loc_iou)
        per_image.append(report)

        summary['images'] += 1
        summary['gt_total'] += len(gts)
        summary['pred_total'] += len(preds)
        summary['tp_total'] += report['tp_count']
        summary['gt_error_counts'].update(report['gt_error_counts'])
        summary['fp_error_counts'].update(report['fp_error_counts'])
        update_class_stats(class_stats, gts, report)

        if idx == 1 or idx % 100 == 0 or idx == len(images):
            print(f'[{idx:05d}/{len(images)}] analyzed {image["file_name"]}')

    summary['gt_error_counts'] = dict(summary['gt_error_counts'])
    summary['fp_error_counts'] = dict(summary['fp_error_counts'])
    summary['recall_at_iou'] = summary['tp_total'] / max(summary['gt_total'], 1)
    summary['precision_at_iou'] = summary['tp_total'] / max(
        summary['pred_total'], 1)
    summary['dominant_error'] = dominant_error(summary)
    summary['per_class_gt_error_counts'] = finalize_class_stats(class_stats)

    detail_examples = []
    for report in per_image:
        if len(detail_examples) >= args.top_error_examples:
            break
        if report['gt_errors'] or report['fp_errors']:
            detail_examples.append(report)

    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / 'error_examples.json', 'w', encoding='utf-8') as f:
        json.dump(detail_examples, f, indent=2)

    image_rows = []
    for report in per_image:
        row = {
            'image_id': report['image_id'],
            'file_name': report['file_name'],
            'gt_count': report['gt_count'],
            'pred_count': report['pred_count'],
            'tp_count': report['tp_count'],
        }
        for key in [
                'classification_error', 'localization_error',
                'classification_and_localization_error', 'missed_background',
                'true_positive'
        ]:
            row[f'gt_{key}'] = report['gt_error_counts'].get(key, 0)
        for key in [
                'classification_fp', 'localization_fp',
                'classification_and_localization_fp', 'background_fp',
                'duplicate_fp'
        ]:
            row[f'fp_{key}'] = report['fp_error_counts'].get(key, 0)
        image_rows.append(row)
    write_csv(out_dir / 'per_image_summary.csv', image_rows,
              list(image_rows[0].keys()) if image_rows else [])

    print(f'Wrote analysis to {out_dir}')
    print(f'Recall@{args.tp_iou:.2f}: {summary["recall_at_iou"]:.4f}')
    print(f'Precision@{args.tp_iou:.2f}: {summary["precision_at_iou"]:.4f}')
    dom = summary['dominant_error']
    print('Dominant GT-side error: '
          f'{dom["dominant_error"]} '
          f'({dom["dominant_error_count"]}, '
          f'{dom["dominant_error_ratio_among_gt_errors"]:.2%} of GT errors)')
    print('Read summary.json for per-class counts and FP-side breakdown.')


if __name__ == '__main__':
    main()
