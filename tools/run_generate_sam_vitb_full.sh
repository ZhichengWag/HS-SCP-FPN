#!/usr/bin/env bash
set -eo pipefail

source /home/wzc/miniconda3/etc/profile.d/conda.sh
conda activate mmdet5090

cd /mnt/e/mmdet5090
mkdir -p work_dirs /mnt/e/AI-TOD/pseudo_labels_sam_vitb/trainval

python mmdetection/tools/generate_scp_sam_pseudo_labels.py \
  --ann-file /mnt/e/AI-TOD/annotations/aitod_trainval_v1.json \
  --img-dir /mnt/e/AI-TOD/trainval/images \
  --output-dir /mnt/e/AI-TOD/pseudo_labels_sam_vitb/trainval \
  --sam-impl sam \
  --sam-checkpoint /mnt/e/models/sam/sam_vit_b_01ec64.pth \
  --sam-model-type vit_b \
  --device cuda:0
