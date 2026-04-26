"""Generate coarse pseudo-labels for SCP distillation.

This script uses SegFormer-B5 (ADE20K pre-trained) to generate per-image
semantic segmentation predictions, then maps ADE20K 150-class labels to
K=8 coarse background classes. Foreground objects (person, car, etc.) are
mapped to ignore_index=255 to avoid task conflict with the detection head.

Usage:
    python generate_pseudo_labels.py \
        --img_dir /path/to/AI-TOD/trainval/images \
        --output_dir /path/to/pseudo_labels \
        --device cuda:0

Requirements:
    pip install transformers pillow numpy tqdm
"""

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    from PIL import Image
    import torch
    import torch.nn.functional as F
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install transformers pillow torch")
    exit(1)


# =============================================================
# ADE20K 150-class → 8 coarse background classes mapping
# =============================================================
# ADE20K class indices (0-indexed, after subtracting 1 from 1-indexed)
# Reference: https://docs.google.com/spreadsheets/d/1se8YEtb2detS7OuPE86fXGyD269pMycAWe2mtKUj2W8

ADE20K_TO_COARSE = {}

# Class 0: Water (sea, river, lake, pool, waterfall)
WATER_IDS = [21, 26, 60, 109, 128, 113, 147]
for idx in WATER_IDS:
    ADE20K_TO_COARSE[idx] = 0

# Class 1: Road / parking / runway
ROAD_IDS = [6, 11, 52, 91, 46, 29]
for idx in ROAD_IDS:
    ADE20K_TO_COARSE[idx] = 1

# Class 2: Building / dense urban
BUILD_IDS = [1, 25, 48, 79, 84, 0]
for idx in BUILD_IDS:
    ADE20K_TO_COARSE[idx] = 2

# Class 3: Airport / runway (specific for airplane detection)
AIRPORT_IDS = [52, 68]
for idx in AIRPORT_IDS:
    ADE20K_TO_COARSE[idx] = 3

# Class 4: Bridge structure
BRIDGE_IDS = [62]
for idx in BRIDGE_IDS:
    ADE20K_TO_COARSE[idx] = 4

# Class 5: Farmland / grassland
FARM_IDS = [13, 29, 81, 9, 14, 16, 94]
for idx in FARM_IDS:
    ADE20K_TO_COARSE[idx] = 5

# Class 6: Vegetation / forest
VEG_IDS = [4, 17, 66, 73, 72]
for idx in VEG_IDS:
    ADE20K_TO_COARSE[idx] = 6

# Foreground classes → 255 (ignore): person, car, truck, bus, boat, airplane, etc.
FOREGROUND_IDS = [12, 20, 83, 80, 76, 90, 116, 127, 102, 19,
                  103, 115, 54, 59, 71, 136]
for idx in FOREGROUND_IDS:
    ADE20K_TO_COARSE[idx] = 255


def map_ade20k_to_coarse(label):
    """Map ADE20K 150-class prediction to 8 coarse classes.
    
    Args:
        label (np.ndarray): (H, W) with values in [0, 149].
    
    Returns:
        np.ndarray: (H, W) with values in {0,...,7, 255}.
    """
    coarse = np.full_like(label, fill_value=7, dtype=np.uint8)  # default: "other"
    
    for ade_id, coarse_id in ADE20K_TO_COARSE.items():
        coarse[label == ade_id] = coarse_id
    
    return coarse


def main():
    parser = argparse.ArgumentParser(
        description='Generate coarse pseudo-labels for SCP distillation')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='Directory containing training images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for pseudo-label .npz files')
    parser.add_argument('--model_name', type=str,
                        default='nvidia/segformer-b5-finetuned-ade-640-640',
                        help='HuggingFace SegFormer model name')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for inference')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    processor = SegformerImageProcessor.from_pretrained(args.model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()

    # Collect all image files
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    img_dir = Path(args.img_dir)
    img_files = sorted([f for f in img_dir.iterdir()
                        if f.suffix.lower() in img_extensions])

    print(f"Found {len(img_files)} images in {args.img_dir}")
    print(f"Saving pseudo-labels to {args.output_dir}")

    skipped = 0
    for img_path in tqdm(img_files, desc="Generating pseudo-labels"):
        output_path = os.path.join(args.output_dir, img_path.stem + '.npz')

        # Skip if already generated
        if os.path.exists(output_path):
            skipped += 1
            continue

        try:
            image = Image.open(img_path).convert('RGB')
            ori_h, ori_w = image.size[1], image.size[0]

            # Process image
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(args.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # (1, num_classes, H/4, W/4)

            # Upsample to original resolution
            logits = F.interpolate(
                logits, size=(ori_h, ori_w),
                mode='bilinear', align_corners=False
            )

            # Get class predictions
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

            # Map to coarse classes
            coarse_label = map_ade20k_to_coarse(pred)

            # Save
            np.savez_compressed(output_path, label=coarse_label)

        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            continue

    print(f"\nDone! Generated {len(img_files) - skipped} pseudo-labels, "
          f"skipped {skipped} existing files.")


if __name__ == '__main__':
    main()
