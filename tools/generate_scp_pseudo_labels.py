"""Generate semantic pseudo-labels for SCP distillation.

Example:
    python tools/generate_scp_pseudo_labels.py \
        --img-dir /mnt/e/AI-TOD/trainval/images \
        --output-dir /mnt/e/AI-TOD/pseudo_labels/trainval \
        --device cuda:0

    # SCPV2 K-class teacher labels, keeping ADE20K's original 150 classes:
    python tools/generate_scp_pseudo_labels.py \
        --img-dir /mnt/e/AI-TOD/trainval/images \
        --output-dir /mnt/e/AI-TOD/pseudo_labels_ade20k/trainval \
        --label-space ade20k \
        --save-confidence \
        --device cuda:0
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


ADE20K_TO_COARSE = {}

for idx in [21, 26, 60, 109, 128, 113, 147]:
    ADE20K_TO_COARSE[idx] = 0
for idx in [6, 11, 52, 91, 46, 29]:
    ADE20K_TO_COARSE[idx] = 1
for idx in [1, 25, 48, 79, 84, 0]:
    ADE20K_TO_COARSE[idx] = 2
for idx in [52, 68]:
    ADE20K_TO_COARSE[idx] = 3
for idx in [62]:
    ADE20K_TO_COARSE[idx] = 4
for idx in [13, 29, 81, 9, 14, 16, 94]:
    ADE20K_TO_COARSE[idx] = 5
for idx in [4, 17, 66, 73, 72]:
    ADE20K_TO_COARSE[idx] = 6
for idx in [12, 20, 83, 80, 76, 90, 116, 127, 102, 19, 103, 115, 54, 59,
            71, 136]:
    ADE20K_TO_COARSE[idx] = 255


def map_ade20k_to_coarse(label):
    coarse = np.full_like(label, fill_value=7, dtype=np.uint8)
    for ade_id, coarse_id in ADE20K_TO_COARSE.items():
        coarse[label == ade_id] = coarse_id
    return coarse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate pseudo-labels for SCP distillation.')
    parser.add_argument('--img-dir', required=True, help='Training image dir.')
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output dir for .npz labels when generating one label space.')
    parser.add_argument(
        '--coarse-output-dir',
        default=None,
        help='Output dir for 8-class labels when --label-space=both.')
    parser.add_argument(
        '--ade20k-output-dir',
        default=None,
        help='Output dir for 150-class ADE20K labels when --label-space=both.')
    parser.add_argument(
        '--model-name',
        default='nvidia/segformer-b5-finetuned-ade-640-640',
        help='HuggingFace model id or a local SegFormer model directory.')
    parser.add_argument(
        '--cache-dir',
        default=None,
        help='Optional HuggingFace cache directory.')
    parser.add_argument(
        '--local-files-only',
        action='store_true',
        help='Only load an already downloaded local/cached model.')
    parser.add_argument(
        '--hf-endpoint',
        default=None,
        help='Optional HuggingFace endpoint, e.g. https://hf-mirror.com.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--label-space',
        choices=('coarse8', 'ade20k', 'both'),
        default='coarse8',
        help=('Pseudo-label category space. "coarse8" keeps the original V1 '
              '8-class mapping; "ade20k" saves raw SegFormer/ADE20K '
              '0..149 class ids for SCPV2 K-class training. "both" runs '
              'the teacher once and writes both outputs.'))
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Number of images per SegFormer forward pass.')
    parser.add_argument(
        '--save-confidence',
        action='store_true',
        help='Save the teacher max-softmax confidence map into each .npz.')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be >= 1.')
    if args.label_space == 'both':
        if args.coarse_output_dir is None or args.ade20k_output_dir is None:
            parser.error('--label-space=both requires --coarse-output-dir and '
                         '--ade20k-output-dir.')
    elif args.output_dir is None:
        parser.error('--output-dir is required unless --label-space=both.')
    return args


def load_segformer(args):
    if args.hf_endpoint:
        os.environ['HF_ENDPOINT'] = args.hf_endpoint

    try:
        from transformers import (SegformerForSemanticSegmentation,
                                  SegformerImageProcessor)
        processor = SegformerImageProcessor.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only)
        model = SegformerForSemanticSegmentation.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only)
    except Exception as exc:
        message = (
            'Failed to load SegFormer model.\n'
            f'  model-name: {args.model_name}\n'
            f'  cache-dir: {args.cache_dir}\n'
            f'  local-files-only: {args.local_files_only}\n'
            '\n'
            'If WSL cannot reach HuggingFace but the model is already cached, '
            'rerun with --local-files-only. For SCPV2 K-class labels:\n'
            '  python tools/generate_scp_pseudo_labels.py '
            '--img-dir /mnt/e/AI-TOD/trainval/images '
            '--output-dir /mnt/e/AI-TOD/pseudo_labels_ade20k/trainval '
            '--label-space ade20k --save-confidence '
            '--local-files-only --device cuda:0\n'
            '\n'
            'If the model is not cached, either use a mirror with '
            '--hf-endpoint https://hf-mirror.com, or download the model once '
            'in an environment with network access, then pass that directory '
            'as --model-name together with --local-files-only. Example:\n'
            '  huggingface-cli download nvidia/segformer-b5-finetuned-ade-640-640 '
            '--local-dir /mnt/e/models/segformer-b5-finetuned-ade-640-640\n'
            '  python tools/generate_scp_pseudo_labels.py '
            '--img-dir /mnt/e/AI-TOD/trainval/images '
            '--output-dir /mnt/e/AI-TOD/pseudo_labels_ade20k/trainval '
            '--model-name /mnt/e/models/segformer-b5-finetuned-ade-640-640 '
            '--label-space ade20k --save-confidence '
            '--local-files-only --device cuda:0')
        raise RuntimeError(message) from exc

    return processor, model


def main():
    args = parse_args()
    output_dirs = {}
    if args.label_space in ('coarse8', 'both'):
        output_dirs['coarse8'] = Path(args.coarse_output_dir or args.output_dir)
    if args.label_space in ('ade20k', 'both'):
        output_dirs['ade20k'] = Path(args.ade20k_output_dir or args.output_dir)
    for output_dir in output_dirs.values():
        output_dir.mkdir(parents=True, exist_ok=True)

    processor, model = load_segformer(args)
    model.to(args.device).eval()

    image_suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_paths = sorted(
        p for p in Path(args.img_dir).iterdir()
        if p.suffix.lower() in image_suffixes)

    skipped = 0
    pending_paths = []
    for image_path in image_paths:
        expected_paths = [
            output_dir / f'{image_path.stem}.npz'
            for output_dir in output_dirs.values()
        ]
        if all(path.exists() for path in expected_paths):
            skipped += 1
            continue
        pending_paths.append(image_path)

    for start in tqdm(
            range(0, len(pending_paths), args.batch_size),
            desc='Generating SCP labels'):
        batch_paths = pending_paths[start:start + args.batch_size]
        images = []
        original_sizes = []
        for image_path in batch_paths:
            image = Image.open(image_path).convert('RGB')
            ori_w, ori_h = image.size
            images.append(image)
            original_sizes.append((ori_h, ori_w))

        inputs = processor(images=images, return_tensors='pt')
        inputs = {key: value.to(args.device) for key, value in inputs.items()}

        with torch.no_grad():
            batch_logits = model(**inputs).logits

        for idx, image_path in enumerate(batch_paths):
            logits = batch_logits[idx:idx + 1]
            ori_h, ori_w = original_sizes[idx]
            logits = F.interpolate(
                logits,
                size=(ori_h, ori_w),
                mode='bilinear',
                align_corners=False)

            probs = None
            if args.save_confidence:
                probs = logits.softmax(dim=1).amax(
                    dim=1).squeeze(0).cpu().numpy()
                probs = probs.astype(np.float16, copy=False)

            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            if 'coarse8' in output_dirs:
                output_path = output_dirs['coarse8'] / f'{image_path.stem}.npz'
                if not output_path.exists():
                    label = map_ade20k_to_coarse(pred)
                    if probs is None:
                        np.savez_compressed(output_path, label=label)
                    else:
                        np.savez_compressed(
                            output_path, label=label, confidence=probs)
            if 'ade20k' in output_dirs:
                output_path = output_dirs['ade20k'] / f'{image_path.stem}.npz'
                if not output_path.exists():
                    label = pred.astype(np.int32, copy=False)
                    if probs is None:
                        np.savez_compressed(output_path, label=label)
                    else:
                        np.savez_compressed(
                            output_path, label=label, confidence=probs)

    print(f'Done. Generated {len(pending_paths)}, skipped {skipped}.')
    print(f'Label space: {args.label_space}.')
    for name, output_dir in output_dirs.items():
        print(f'{name} output dir: {output_dir}')


if __name__ == '__main__':
    main()
