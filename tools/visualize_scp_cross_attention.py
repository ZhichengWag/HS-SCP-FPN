"""Visualize HS-SCP-FPN semantic cross-attention heatmaps.

This script does not modify the model source.  It monkey-patches the runtime
``SemanticCrossAttention`` modules to cache compact attention heatmaps during
normal MMDetection inference, then renders random image-level visualizations.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from types import MethodType

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
DEFAULT_SCP_CLASS_NAMES = (
    'water',
    'road/runway',
    'building',
    'airport',
    'bridge',
    'farmland',
    'vegetation',
    'other',
)

LANCZOS = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS')
BILINEAR = getattr(getattr(Image, 'Resampling', Image), 'BILINEAR')
NEAREST = getattr(getattr(Image, 'Resampling', Image), 'NEAREST')


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
        description='Visualize HFP-to-SCP cross-attention heatmaps.')
    parser.add_argument(
        '--config',
        default=str(WORKSPACE_ROOT / 'config_hsfpn' /
                    'cascade_rcnn_r50_aitod_scp.py'),
        help='HS-SCP-FPN config path.')
    parser.add_argument(
        '--checkpoint',
        default=str(WORKSPACE_ROOT / 'work_dirs' /
                    'cascade_rcnn_r50_aitod_scp_b1_0.005' /
                    'best_coco_bbox_mAP_epoch_12.pth'),
        help='Checkpoint path.')
    parser.add_argument(
        '--img-dir',
        default=default_aitod_img_dir(),
        help='Directory to randomly sample images from.')
    parser.add_argument(
        '--out-dir',
        default=str(WORKSPACE_ROOT / 'visualizations' /
                    'scp_cross_attention_100'),
        help='Directory to write heatmap visualizations.')
    parser.add_argument('--sample-n', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260603)
    parser.add_argument(
        '--device',
        default='auto',
        help='Inference device. Use "auto", "cuda:0", or "cpu".')
    parser.add_argument(
        '--attn-reduce',
        choices=('mean', 'max'),
        default='mean',
        help='Reduce query-to-key attention as a spatial SCP-key heatmap.')
    parser.add_argument('--alpha', type=float, default=0.48)
    parser.add_argument('--panel-size', type=int, default=300)
    parser.add_argument('--thumb-size', type=int, default=220)
    parser.add_argument(
        '--class-names',
        default=','.join(DEFAULT_SCP_CLASS_NAMES),
        help='Comma-separated SCP semantic channel names.')
    parser.add_argument(
        '--class-top-k',
        type=int,
        default=5,
        help='Number of semantic channels shown in the class-attention panel.')
    parser.add_argument(
        '--channel-map-top-k',
        type=int,
        default=3,
        help='Number of top SCP semantic channels rendered as heatmaps.')
    parser.add_argument(
        '--no-channel-heatmaps',
        action='store_true',
        help='Disable per-semantic-channel attention heatmap panels.')
    parser.add_argument(
        '--gt-query-channel-top-k',
        type=int,
        default=3,
        help='Number of top GT-query semantic channels rendered as heatmaps.')
    parser.add_argument(
        '--no-gt-query-panels',
        action='store_true',
        help='Disable GT-box query attention and channel-score panels.')
    parser.add_argument(
        '--feature-levels',
        default='P1,P2',
        help='Comma-separated levels for pre/post SCP feature visualization.')
    parser.add_argument(
        '--no-feature-panels',
        action='store_true',
        help='Disable P-level pre/post SCP feature and frequency panels.')
    parser.add_argument(
        '--feature-channel-top-k',
        type=int,
        default=2,
        help='Number of top changed feature channels shown per P-level.')
    parser.add_argument(
        '--no-feature-energy-panels',
        action='store_true',
        help='Hide channel-RMS feature energy panels and keep channel maps.')
    parser.add_argument(
        '--freq-low-ratio',
        type=float,
        default=0.25,
        help='Normalized radial FFT threshold for low-frequency energy.')
    parser.add_argument(
        '--freq-high-ratio',
        type=float,
        default=0.50,
        help='Normalized radial FFT threshold for high-frequency energy.')
    parser.add_argument(
        '--no-gt-boxes',
        action='store_true',
        help='Do not draw ground-truth boxes from the test COCO annotation.')
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


def parse_class_names(names):
    parsed = [name.strip() for name in str(names).split(',') if name.strip()]
    return parsed or list(DEFAULT_SCP_CLASS_NAMES)


def parse_levels(levels):
    parsed = [level.strip().upper() for level in str(levels).split(',')
              if level.strip()]
    return parsed or ['P1', 'P2']


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


def build_feature_gt_query_mask(original_size, gt_boxes, feature_size, device,
                                batch_size):
    if not gt_boxes or original_size is None:
        return None
    ori_w, ori_h = original_size
    feat_h, feat_w = feature_size
    mask_img = Image.new('L', (feat_w, feat_h), 0)
    draw = ImageDraw.Draw(mask_img)
    for gt in gt_boxes:
        x, y, w, h = gt['bbox']
        x1 = int(math.floor(x * feat_w / max(ori_w, 1)))
        y1 = int(math.floor(y * feat_h / max(ori_h, 1)))
        x2 = int(math.ceil((x + w) * feat_w / max(ori_w, 1))) - 1
        y2 = int(math.ceil((y + h) * feat_h / max(ori_h, 1))) - 1
        x1 = max(0, min(feat_w - 1, x1))
        y1 = max(0, min(feat_h - 1, y1))
        x2 = max(x1, min(feat_w - 1, x2))
        y2 = max(y1, min(feat_h - 1, y2))
        draw.rectangle((x1, y1, x2, y2), fill=255)
    mask = np.asarray(mask_img, dtype=np.uint8) > 0
    mask = torch.from_numpy(mask).to(device=device, dtype=torch.bool)
    return mask.unsqueeze(0).expand(batch_size, -1, -1)


def feature_energy_heat(feature):
    return feature.detach().float().square().mean(
        dim=1, keepdim=True).sqrt()


def select_top_changed_channels(pre_feature, post_feature, top_k):
    delta = (post_feature[:1] - pre_feature[:1]).detach().float()
    scores = delta.abs().mean(dim=(0, 2, 3))
    top_k = min(max(int(top_k), 0), scores.numel())
    if top_k <= 0:
        return [], []
    values, indices = torch.topk(scores, k=top_k)
    return [int(i) for i in indices.cpu().tolist()], [
        float(v) for v in values.cpu().tolist()
    ]


def normalize_signed_feature_channel(channel):
    channel = np.asarray(channel, dtype=np.float32)
    finite = np.isfinite(channel)
    if not finite.any():
        return np.zeros_like(channel, dtype=np.float32)
    values = channel[finite]
    center = float(np.median(values))
    spread = np.percentile(np.abs(values - center), 98)
    if spread <= 1e-12:
        spread = float(np.max(np.abs(values - center)))
    if spread <= 1e-12:
        return np.zeros_like(channel, dtype=np.float32)
    out = (channel - center) / (2.0 * spread) + 0.5
    out = np.clip(out, 0.0, 1.0)
    out[~finite] = 0.5
    return out.astype(np.float32, copy=False)


def feature_frequency_stats(feature, low_ratio, high_ratio):
    x = feature[:1].detach().float()
    _, _, h, w = x.shape
    x = x - x.mean(dim=(2, 3), keepdim=True)
    spectrum = torch.fft.fft2(x, dim=(2, 3), norm='ortho')
    power = spectrum.abs().square().mean(dim=(0, 1))

    fy = torch.fft.fftfreq(h, device=power.device).view(h, 1)
    fx = torch.fft.fftfreq(w, device=power.device).view(1, w)
    radius = torch.sqrt(fy.square() + fx.square())
    radius = radius / math.sqrt(0.5**2 + 0.5**2)

    total = power.sum().clamp_min(1e-12)
    low_mask = radius <= float(low_ratio)
    high_mask = radius >= float(high_ratio)
    mid_mask = ~(low_mask | high_mask)
    low_energy = power[low_mask].sum()
    mid_energy = power[mid_mask].sum()
    high_energy = power[high_mask].sum()
    centroid = (power * radius).sum() / total

    return {
        'shape': [int(h), int(w)],
        'zero_centered': True,
        'total_energy': float(total.detach().cpu()),
        'low_ratio': float((low_energy / total).detach().cpu()),
        'mid_ratio': float((mid_energy / total).detach().cpu()),
        'high_ratio': float((high_energy / total).detach().cpu()),
        'frequency_centroid': float(centroid.detach().cpu()),
        'spatial_rms': float(x.square().mean().sqrt().detach().cpu()),
        'mean_abs': float(x.abs().mean().detach().cpu()),
    }


def frequency_change_stats(pre_stats, post_stats):
    if not pre_stats or not post_stats:
        return None
    delta_high = post_stats['high_ratio'] - pre_stats['high_ratio']
    delta_low = post_stats['low_ratio'] - pre_stats['low_ratio']
    delta_centroid = (post_stats['frequency_centroid'] -
                      pre_stats['frequency_centroid'])
    if delta_high > 1e-4 and delta_centroid > 0:
        direction = 'higher-frequency'
    elif delta_high < -1e-4 and delta_centroid < 0:
        direction = 'lower-frequency'
    else:
        direction = 'mixed/flat'
    return {
        'delta_low_ratio': float(delta_low),
        'delta_high_ratio': float(delta_high),
        'delta_frequency_centroid': float(delta_centroid),
        'direction': direction,
    }


def patch_semantic_cross_attention(model, attn_reduce, feature_levels,
                                   freq_low_ratio, freq_high_ratio,
                                   feature_channel_top_k):
    patched = []
    feature_levels = set(feature_levels)

    def forward_with_capture(self, hfp_feat, scp_feat, patch_size):
        bsz, _, h, w = hfp_feat.shape
        ph, pw = int(patch_size[0]), int(patch_size[1])
        gh, gw = h // ph, w // pw

        q = self.conv_q(hfp_feat)
        k = self.conv_k(scp_feat)
        v = self.conv_v(scp_feat)

        q = q.view(bsz, self.attn_dim, gh, ph, gw, pw).permute(
            0, 2, 4, 3, 5, 1).reshape(bsz * gh * gw, ph * pw,
                                       self.attn_dim)
        k = k.view(bsz, self.attn_dim, gh, ph, gw, pw).permute(
            0, 2, 4, 1, 3, 5).reshape(bsz * gh * gw, self.attn_dim,
                                      ph * pw)
        v = v.view(bsz, self.attn_dim, gh, ph, gw, pw).permute(
            0, 2, 4, 3, 5, 1).reshape(bsz * gh * gw, ph * pw,
                                       self.attn_dim)

        attn = torch.matmul(q, k) / math.sqrt(self.attn_dim)
        attn = self.softmax(attn)
        out = torch.matmul(attn, v)
        out = out.view(bsz, gh, gw, ph, pw, self.attn_dim).permute(
            0, 5, 1, 3, 2, 4).reshape(bsz, self.attn_dim, h, w)
        fused = hfp_feat + self.out_proj(out)

        with torch.no_grad():
            level = getattr(self, '_scp_vis_level', '')
            attn_float = attn.detach().float()
            if attn_reduce == 'max':
                key_heat = attn_float.amax(dim=1)
            else:
                key_heat = attn_float.mean(dim=1)

            key_heat = key_heat.view(bsz, gh, gw, ph, pw).permute(
                0, 1, 3, 2, 4).reshape(bsz, 1, h, w)
            hfp_heat = feature_energy_heat(hfp_feat)
            scp_prob = scp_feat.detach().float().softmax(dim=1)
            scp_heat = scp_prob.amax(dim=1, keepdim=True)
            channel_attn = scp_prob * key_heat
            class_numer = (scp_prob * key_heat).sum(dim=(2, 3))
            class_denom = key_heat.sum(dim=(2, 3)).clamp_min(1e-6)
            class_scores = class_numer / class_denom
            class_presence = scp_prob.mean(dim=(2, 3))
            gt_key_heat = None
            gt_channel_attn = None
            gt_class_scores = None
            gt_query_class_scores = None
            gt_query_count = None
            feature_payload = None

            gt_mask = build_feature_gt_query_mask(
                getattr(self, '_scp_vis_original_size', None),
                getattr(self, '_scp_vis_gt_boxes', None), (h, w),
                attn_float.device, bsz)
            if gt_mask is not None and gt_mask.any():
                query_mask = gt_mask.view(bsz, gh, ph, gw, pw).permute(
                    0, 1, 3, 2, 4).reshape(bsz * gh * gw, ph * pw)
                query_weight = query_mask.float()
                has_query = query_weight.sum(dim=1, keepdim=True) > 0

                if attn_reduce == 'max':
                    masked_attn = attn_float.masked_fill(
                        ~query_mask.unsqueeze(-1), -float('inf'))
                    gt_key_patch = masked_attn.amax(dim=1)
                    gt_key_patch = torch.where(
                        has_query,
                        gt_key_patch,
                        torch.zeros_like(gt_key_patch))
                else:
                    denom = query_weight.sum(dim=1, keepdim=True).clamp_min(
                        1e-6)
                    gt_key_patch = (
                        attn_float *
                        query_weight.unsqueeze(-1)).sum(dim=1) / denom
                    gt_key_patch = torch.where(
                        has_query,
                        gt_key_patch,
                        torch.zeros_like(gt_key_patch))

                gt_key_heat = gt_key_patch.view(bsz, gh, gw, ph, pw).permute(
                    0, 1, 3, 2, 4).reshape(bsz, 1, h, w)
                gt_channel_attn = scp_prob * gt_key_heat
                gt_class_numer = (scp_prob * gt_key_heat).sum(dim=(2, 3))
                gt_class_denom = gt_key_heat.sum(dim=(2, 3)).clamp_min(1e-6)
                gt_class_scores = gt_class_numer / gt_class_denom

                scp_prob_keys = scp_prob.view(
                    bsz, scp_prob.shape[1], gh, ph, gw, pw).permute(
                        0, 2, 4, 3, 5, 1).reshape(bsz * gh * gw, ph * pw,
                                                   scp_prob.shape[1])
                query_class = torch.matmul(attn_float, scp_prob_keys)
                query_class = query_class.view(bsz, gh * gw, ph * pw,
                                               scp_prob.shape[1])
                query_weight_b = query_weight.view(bsz, gh * gw, ph * pw)
                gt_query_count = query_weight_b.sum(dim=(1, 2))
                gt_query_class_scores = (
                    query_class *
                    query_weight_b.unsqueeze(-1)).sum(dim=(1, 2))
                gt_query_class_scores = gt_query_class_scores / (
                    gt_query_count.view(bsz, 1).clamp_min(1e-6))

            if level in feature_levels:
                pre_heat = feature_energy_heat(hfp_feat)
                post_heat = feature_energy_heat(fused)
                delta_heat = feature_energy_heat(fused - hfp_feat)
                top_channels, top_channel_scores = select_top_changed_channels(
                    hfp_feat, fused, feature_channel_top_k)
                pre_channels = (hfp_feat[:1, top_channels].detach().float().
                                cpu() if top_channels else None)
                post_channels = (fused[:1, top_channels].detach().float().
                                 cpu() if top_channels else None)
                delta_channels = ((fused - hfp_feat)[:1, top_channels].
                                  detach().float().cpu()
                                  if top_channels else None)
                pre_freq = feature_frequency_stats(hfp_feat, freq_low_ratio,
                                                   freq_high_ratio)
                post_freq = feature_frequency_stats(fused, freq_low_ratio,
                                                    freq_high_ratio)
                feature_payload = {
                    'pre_heat': pre_heat[:1].cpu(),
                    'post_heat': post_heat[:1].cpu(),
                    'delta_heat': delta_heat[:1].cpu(),
                    'top_channels': top_channels,
                    'top_channel_scores': top_channel_scores,
                    'pre_channels': pre_channels,
                    'post_channels': post_channels,
                    'delta_channels': delta_channels,
                    'pre_frequency': pre_freq,
                    'post_frequency': post_freq,
                    'frequency_change':
                    frequency_change_stats(pre_freq, post_freq),
                    'freq_low_ratio': float(freq_low_ratio),
                    'freq_high_ratio': float(freq_high_ratio),
                }

            self._scp_vis = {
                'name': getattr(self, '_scp_vis_name', ''),
                'level': level,
                'patch_size': [ph, pw],
                'attn': key_heat[:1].cpu(),
                'hfp': hfp_heat[:1].cpu(),
                'scp': scp_heat[:1].cpu(),
                'scp_prob': scp_prob[:1].cpu(),
                'channel_attn': channel_attn[:1].cpu(),
                'class_scores': class_scores[:1].cpu(),
                'class_presence': class_presence[:1].cpu(),
                'gt_attn':
                gt_key_heat[:1].cpu() if gt_key_heat is not None else None,
                'gt_channel_attn':
                gt_channel_attn[:1].cpu()
                if gt_channel_attn is not None else None,
                'gt_class_scores':
                gt_class_scores[:1].cpu()
                if gt_class_scores is not None else None,
                'gt_query_class_scores':
                gt_query_class_scores[:1].cpu()
                if gt_query_class_scores is not None else None,
                'gt_query_count':
                gt_query_count[:1].cpu()
                if gt_query_count is not None else None,
                'feature':
                feature_payload,
            }

        return fused

    for name, module in model.named_modules():
        if module.__class__.__name__ != 'SemanticCrossAttention':
            continue
        module._scp_vis_name = name
        module._scp_vis_level = infer_level(name)
        module._scp_vis_feature_levels = feature_levels
        module._scp_vis = None
        module._scp_vis_gt_boxes = None
        module._scp_vis_original_size = None
        module._scp_vis_original_forward = module.forward
        module.forward = MethodType(forward_with_capture, module)
        patched.append(module)

    return patched


def clear_cached_attention(patched_modules):
    for module in patched_modules:
        module._scp_vis = None


def set_gt_query_context(patched_modules, original_size, gt_boxes):
    for module in patched_modules:
        module._scp_vis_original_size = original_size
        module._scp_vis_gt_boxes = gt_boxes


def collect_attention(model):
    items = []
    for _, module in model.named_modules():
        payload = getattr(module, '_scp_vis', None)
        if not payload:
            continue
        attn = payload['attn'][0, 0].numpy()
        hfp = payload['hfp'][0, 0].numpy()
        scp = payload['scp'][0, 0].numpy()
        scp_prob = payload['scp_prob'][0].numpy()
        channel_attn = payload['channel_attn'][0].numpy()
        class_scores = payload['class_scores'][0].numpy()
        class_presence = payload['class_presence'][0].numpy()
        gt_attn = payload.get('gt_attn')
        gt_channel_attn = payload.get('gt_channel_attn')
        gt_class_scores = payload.get('gt_class_scores')
        gt_query_class_scores = payload.get('gt_query_class_scores')
        gt_query_count = payload.get('gt_query_count')
        feature_payload = payload.get('feature')
        feature = None
        if feature_payload is not None:
            pre_channels = feature_payload.get('pre_channels')
            post_channels = feature_payload.get('post_channels')
            delta_channels = feature_payload.get('delta_channels')
            feature = {
                'pre_heat':
                feature_payload['pre_heat'][0, 0].numpy(),
                'post_heat':
                feature_payload['post_heat'][0, 0].numpy(),
                'delta_heat':
                feature_payload['delta_heat'][0, 0].numpy(),
                'top_channels':
                feature_payload['top_channels'],
                'top_channel_scores':
                feature_payload['top_channel_scores'],
                'pre_channels':
                pre_channels[0].numpy() if pre_channels is not None else None,
                'post_channels':
                post_channels[0].numpy()
                if post_channels is not None else None,
                'delta_channels':
                delta_channels[0].numpy()
                if delta_channels is not None else None,
                'pre_frequency':
                feature_payload['pre_frequency'],
                'post_frequency':
                feature_payload['post_frequency'],
                'frequency_change':
                feature_payload['frequency_change'],
                'freq_low_ratio':
                feature_payload['freq_low_ratio'],
                'freq_high_ratio':
                feature_payload['freq_high_ratio'],
            }
        items.append({
            'name': payload['name'],
            'level': payload['level'],
            'patch_size': payload['patch_size'],
            'attn': attn,
            'hfp': hfp,
            'scp': scp,
            'scp_prob': scp_prob,
            'channel_attn': channel_attn,
            'class_scores': class_scores,
            'class_presence': class_presence,
            'gt_attn':
            gt_attn[0, 0].numpy() if gt_attn is not None else None,
            'gt_channel_attn':
            gt_channel_attn[0].numpy()
            if gt_channel_attn is not None else None,
            'gt_class_scores':
            gt_class_scores[0].numpy()
            if gt_class_scores is not None else None,
            'gt_query_class_scores':
            gt_query_class_scores[0].numpy()
            if gt_query_class_scores is not None else None,
            'gt_query_count':
            float(gt_query_count[0].item())
            if gt_query_count is not None else 0.0,
            'feature':
            feature,
            'attn_shape': list(attn.shape),
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


def resize_float_to_original(value, meta, level, original_size):
    ori_w, ori_h = original_size
    value = np.asarray(value, dtype=np.float32)
    img_shape = meta.get('img_shape', None)
    if img_shape is None:
        out = Image.fromarray(value)
        return np.asarray(out.resize((ori_w, ori_h), BILINEAR),
                          dtype=np.float32)

    img_h, img_w = int(img_shape[0]), int(img_shape[1])
    stride = LEVEL_STRIDES.get(level)
    if stride is None:
        pad_h, pad_w = img_h, img_w
    else:
        pad_h, pad_w = value.shape[0] * stride, value.shape[1] * stride

    value_img = Image.fromarray(value)
    value_img = value_img.resize((pad_w, pad_h), BILINEAR)
    value_img = value_img.crop((0, 0, min(img_w, pad_w), min(img_h, pad_h)))
    if value_img.size != (img_w, img_h):
        value_img = value_img.resize((img_w, img_h), BILINEAR)
    value_img = value_img.resize((ori_w, ori_h), BILINEAR)
    return np.asarray(value_img, dtype=np.float32)


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


def build_gt_category_masks(size, gt_boxes):
    width, height = size
    masks = {}
    counts = {}
    for gt in gt_boxes:
        name = gt.get('category_name') or str(gt.get('category_id', ''))
        if not name:
            continue
        mask_img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        x, y, w, h = gt['bbox']
        draw.rectangle((x, y, x + w, y + h), fill=255)
        mask = np.asarray(mask_img, dtype=bool)
        masks[name] = mask if name not in masks else masks[name] | mask
        counts[name] = counts.get(name, 0) + 1
    return masks, counts


def channel_gt_association(item, meta, original_size, gt_boxes, class_names):
    channel_attn = np.asarray(item['channel_attn'], dtype=np.float32)
    scores = np.asarray(item['class_scores'], dtype=np.float32)
    presence = np.asarray(item['class_presence'], dtype=np.float32)
    gt_mask = build_gt_mask(original_size, gt_boxes) if gt_boxes else None
    cat_masks, cat_counts = build_gt_category_masks(
        original_size, gt_boxes) if gt_boxes else ({}, {})
    stats = []

    for ch in range(channel_attn.shape[0]):
        heat = resize_float_to_original(channel_attn[ch], meta,
                                        item['level'], original_size)
        item_stats = {
            'channel':
            int(ch),
            'class_name':
            class_label(int(ch), class_names),
            'attention_score':
            float(scores[ch]),
            'semantic_presence':
            float(presence[ch]),
            'attention_minus_presence':
            float(scores[ch] - presence[ch]),
            'channel_attention_mean':
            float(heat.mean()),
            'inside_gt_mean':
            None,
            'outside_gt_mean':
            None,
            'gt_focus_ratio':
            None,
            'top_gt_categories':
            [],
        }

        if gt_mask is not None and gt_mask.any() and (~gt_mask).any():
            inside = float(heat[gt_mask].mean())
            outside = float(heat[~gt_mask].mean())
            item_stats['inside_gt_mean'] = inside
            item_stats['outside_gt_mean'] = outside
            item_stats['gt_focus_ratio'] = inside / max(outside, 1e-12)

            cat_focus = []
            for name, mask in cat_masks.items():
                if not mask.any():
                    continue
                cat_focus.append({
                    'category_name': name,
                    'box_count': int(cat_counts.get(name, 0)),
                    'inside_mean': float(heat[mask].mean()),
                })
            cat_focus.sort(key=lambda x: x['inside_mean'], reverse=True)
            item_stats['top_gt_categories'] = cat_focus[:3]

        stats.append(item_stats)

    stats.sort(key=lambda x: x['attention_score'], reverse=True)
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
            shadow = [
                rect[0] - delta,
                rect[1] - delta,
                rect[2] + delta,
                rect[3] + delta,
            ]
            draw.rectangle(shadow, outline=(0, 0, 0))
        for delta in range(line_width):
            box = [
                rect[0] - delta,
                rect[1] - delta,
                rect[2] + delta,
                rect[3] + delta,
            ]
            draw.rectangle(box, outline=(255, 230, 0))


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


def class_label(index, class_names):
    if index < len(class_names):
        return class_names[index]
    return f'class_{index}'


def truncate_text(text, max_chars):
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max(1, max_chars - 1)] + '~'


def make_class_attention_panel(attn_items, class_names, top_k, panel_size,
                               font):
    title_h = 24
    panel = Image.new('RGB', (panel_size, panel_size + title_h),
                      (245, 245, 242))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel_size, title_h), fill=(34, 34, 34))
    draw.text((8, 5), 'SCP channel attention', fill=(245, 245, 245),
              font=font)

    y = title_h + 8
    for item in attn_items:
        scores = np.asarray(item['class_scores'], dtype=np.float32)
        presence = np.asarray(item['class_presence'], dtype=np.float32)
        top_n = min(max(int(top_k), 1), len(scores))
        order = np.argsort(scores)[::-1][:top_n]
        max_score = max(float(scores[order[0]]), 1e-6)

        draw.text((10, y), f'{item["level"]}', fill=(20, 20, 20), font=font)
        y += 16
        for idx in order:
            if y > panel.height - 18:
                return panel
            label = class_label(int(idx), class_names)
            label = label[:13]
            bar_w = int((panel_size - 128) * float(scores[idx]) / max_score)
            draw.rectangle((96, y + 2, 96 + bar_w, y + 13),
                           fill=(220, 36, 48))
            draw.text((12, y), f'{idx}:{label}', fill=(20, 20, 20),
                      font=font)
            draw.text((100 + bar_w, y),
                      f'{scores[idx]:.2f}/{presence[idx]:.2f}',
                      fill=(20, 20, 20),
                      font=font)
            y += 15
        y += 5
    return panel


def make_channel_attention_panels(original, attn_items, meta, original_size,
                                  alpha, panel_size, font, gt_boxes,
                                  gt_line_width, class_names, top_k):
    panels = []
    top_k = max(int(top_k), 0)
    if top_k <= 0:
        return panels

    for item in attn_items:
        scores = np.asarray(item['class_scores'], dtype=np.float32)
        channel_attn = np.asarray(item['channel_attn'], dtype=np.float32)
        top_n = min(top_k, channel_attn.shape[0], scores.shape[0])
        order = np.argsort(scores)[::-1][:top_n]
        for ch in order:
            heat = resize_float_to_original(channel_attn[int(ch)], meta,
                                            item['level'], original_size)
            label = truncate_text(class_label(int(ch), class_names), 12)
            overlay = overlay_heatmap(original, heat, alpha)
            panels.append(
                make_panel(overlay,
                           f'{item["level"]} ch{int(ch)} {label}',
                           panel_size,
                           font,
                           gt_boxes=gt_boxes,
                           gt_line_width=gt_line_width))
    return panels


def make_gt_query_attention_panels(original, attn_items, meta, original_size,
                                   alpha, panel_size, font, gt_boxes,
                                   gt_line_width):
    panels = []
    for item in attn_items:
        if item.get('gt_attn') is None or item.get('gt_query_count', 0) <= 0:
            continue
        heat = resize_float_to_original(item['gt_attn'], meta, item['level'],
                                        original_size)
        overlay = overlay_heatmap(original, heat, alpha)
        panels.append(
            make_panel(overlay,
                       f'{item["level"]} GT-Q->SCP attn',
                       panel_size,
                       font,
                       gt_boxes=gt_boxes,
                       gt_line_width=gt_line_width))
    return panels


def make_gt_query_channel_score_panel(attn_items, class_names, top_k,
                                      panel_size, font):
    title_h = 24
    panel = Image.new('RGB', (panel_size, panel_size + title_h),
                      (245, 245, 242))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel_size, title_h), fill=(34, 34, 34))
    draw.text((8, 5), 'GT query channel scores', fill=(245, 245, 245),
              font=font)

    y = title_h + 8
    any_rows = False
    for item in attn_items:
        scores = item.get('gt_query_class_scores')
        if scores is None or item.get('gt_query_count', 0) <= 0:
            continue
        scores = np.asarray(scores, dtype=np.float32)
        top_n = min(max(int(top_k), 1), len(scores))
        order = np.argsort(scores)[::-1][:top_n]
        max_score = max(float(scores[order[0]]), 1e-6)

        any_rows = True
        draw.text((10, y), f'{item["level"]} q={item["gt_query_count"]:.0f}',
                  fill=(20, 20, 20), font=font)
        y += 16
        for idx in order:
            if y > panel.height - 18:
                return panel
            label = truncate_text(class_label(int(idx), class_names), 12)
            bar_w = int((panel_size - 128) * float(scores[idx]) / max_score)
            draw.rectangle((96, y + 2, 96 + bar_w, y + 13),
                           fill=(220, 36, 48))
            draw.text((12, y), f'{idx}:{label}', fill=(20, 20, 20),
                      font=font)
            draw.text((100 + bar_w, y), f'{scores[idx]:.3f}',
                      fill=(20, 20, 20), font=font)
            y += 15
        y += 5

    if not any_rows:
        draw.text((12, title_h + 14), 'No GT query pixels',
                  fill=(30, 30, 30), font=font)
    return panel


def make_gt_query_channel_panels(original, attn_items, meta, original_size,
                                 alpha, panel_size, font, gt_boxes,
                                 gt_line_width, class_names, top_k):
    panels = []
    top_k = max(int(top_k), 0)
    if top_k <= 0:
        return panels

    for item in attn_items:
        channel_attn = item.get('gt_channel_attn')
        scores = item.get('gt_query_class_scores')
        if (channel_attn is None or scores is None or
                item.get('gt_query_count', 0) <= 0):
            continue
        channel_attn = np.asarray(channel_attn, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        top_n = min(top_k, channel_attn.shape[0], scores.shape[0])
        order = np.argsort(scores)[::-1][:top_n]
        for ch in order:
            heat = resize_float_to_original(channel_attn[int(ch)], meta,
                                            item['level'], original_size)
            label = truncate_text(class_label(int(ch), class_names), 10)
            overlay = overlay_heatmap(original, heat, alpha)
            panels.append(
                make_panel(overlay,
                           f'{item["level"]} GT-Q ch{int(ch)} {label}',
                           panel_size,
                           font,
                           gt_boxes=gt_boxes,
                           gt_line_width=gt_line_width))
    return panels


def make_feature_prepost_panels(original, attn_items, meta, original_size,
                                alpha, panel_size, font, gt_boxes,
                                gt_line_width, draw_energy_panels):
    panels = []
    for item in attn_items:
        feature = item.get('feature')
        if feature is None:
            continue
        if draw_energy_panels:
            for key, title in (('pre_heat', 'pre-SCP RMS'),
                               ('post_heat', 'post-SCP RMS'),
                               ('delta_heat', 'SCP delta RMS')):
                heat = resize_float_to_original(feature[key], meta,
                                                item['level'], original_size)
                overlay = overlay_heatmap(original, heat, alpha)
                panels.append(
                    make_panel(overlay,
                               f'{item["level"]} {title}',
                               panel_size,
                               font,
                               gt_boxes=gt_boxes,
                               gt_line_width=gt_line_width))

        top_channels = feature.get('top_channels') or []
        pre_channels = feature.get('pre_channels')
        post_channels = feature.get('post_channels')
        delta_channels = feature.get('delta_channels')
        if pre_channels is None or post_channels is None:
            continue
        for local_idx, channel_idx in enumerate(top_channels):
            for array_key, title in (('pre_channels', 'pre-SCP fmap'),
                                     ('post_channels', 'post-SCP fmap'),
                                     ('delta_channels', 'delta fmap')):
                array = feature.get(array_key)
                if array is None:
                    continue
                heat = normalize_signed_feature_channel(array[local_idx])
                heat = resize_heat_to_original(heat, meta, item['level'],
                                               original_size)
                overlay = overlay_heatmap(original, heat, alpha)
                panels.append(
                    make_panel(overlay,
                               f'{item["level"]} {title} ch{channel_idx}',
                               panel_size,
                               font,
                               gt_boxes=gt_boxes,
                               gt_line_width=gt_line_width))
    return panels


def make_frequency_panel(attn_items, panel_size, font):
    title_h = 24
    panel = Image.new('RGB', (panel_size, panel_size + title_h),
                      (245, 245, 242))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel_size, title_h), fill=(34, 34, 34))
    draw.text((8, 5), 'P1/P2 feature frequency', fill=(245, 245, 245),
              font=font)

    rows = []
    for item in attn_items:
        feature = item.get('feature')
        if feature is None:
            continue
        pre = feature['pre_frequency']
        post = feature['post_frequency']
        change = feature['frequency_change']
        rows.append((item['level'], 'pre', pre['low_ratio'],
                     pre['high_ratio'], pre['frequency_centroid'], None))
        rows.append((item['level'], 'post', post['low_ratio'],
                     post['high_ratio'], post['frequency_centroid'],
                     change))

    if not rows:
        draw.text((12, title_h + 14), 'No feature stats',
                  fill=(30, 30, 30), font=font)
        return panel

    y = title_h + 8
    for level, stage, low, high, centroid, change in rows:
        if y > panel.height - 34:
            break
        label = f'{level} {stage}'
        draw.text((10, y), label, fill=(20, 20, 20), font=font)
        draw.text((76, y), f'L {low:.2f}', fill=(20, 20, 20), font=font)
        draw.text((128, y), f'H {high:.2f}', fill=(20, 20, 20), font=font)
        draw.text((180, y), f'C {centroid:.2f}', fill=(20, 20, 20),
                  font=font)
        y += 14

        low_w = int((panel_size - 28) * np.clip(low, 0.0, 1.0))
        high_w = int((panel_size - 28) * np.clip(high, 0.0, 1.0))
        draw.rectangle((12, y + 1, 12 + low_w, y + 5),
                       fill=(28, 145, 212))
        draw.rectangle((12, y + 7, 12 + high_w, y + 11),
                       fill=(220, 36, 48))
        y += 15

        if change is not None:
            direction = truncate_text(change['direction'], 16)
            delta_high = change['delta_high_ratio']
            delta_centroid = change['delta_frequency_centroid']
            draw.text((16, y),
                      f'dH {delta_high:+.3f} dC {delta_centroid:+.3f} {direction}',
                      fill=(20, 20, 20),
                      font=font)
            y += 16
        y += 4
    return panel


def render_visualization(image_path, attn_items, meta, out_path, alpha,
                         panel_size, gt_boxes, gt_line_width, class_names,
                         class_top_k, channel_map_top_k,
                         draw_channel_heatmaps, gt_query_channel_top_k,
                         draw_gt_query_panels, draw_feature_panels,
                         draw_feature_energy_panels):
    original = Image.open(image_path).convert('RGB')
    original_size = original.size
    font = ImageFont.load_default()

    gt_title = f'Original + GT ({len(gt_boxes)})' if gt_boxes else 'Original'
    panels = [
        make_panel(original, gt_title, panel_size, font, gt_boxes,
                   gt_line_width)
    ]
    aligned_attn = []

    for item in attn_items:
        heat = normalize_heatmap(item['attn'])
        heat = resize_heat_to_original(heat, meta, item['level'],
                                       original_size)
        aligned_attn.append(heat)
        overlay = overlay_heatmap(original, heat, alpha)
        panels.append(
            make_panel(overlay, f'{item["level"]} HFP->SCP attn', panel_size,
                       font))

    if aligned_attn:
        mean_heat = np.mean(np.stack(aligned_attn, axis=0), axis=0)
        panels.append(
            make_panel(overlay_heatmap(original, mean_heat, alpha),
                       'Mean attention', panel_size, font))

    panels.append(
        make_class_attention_panel(attn_items, class_names, class_top_k,
                                   panel_size, font))

    if draw_channel_heatmaps:
        panels.extend(
            make_channel_attention_panels(original, attn_items, meta,
                                          original_size, alpha, panel_size,
                                          font, gt_boxes, gt_line_width,
                                          class_names, channel_map_top_k))

    if draw_gt_query_panels and gt_boxes:
        panels.extend(
            make_gt_query_attention_panels(original, attn_items, meta,
                                           original_size, alpha, panel_size,
                                           font, gt_boxes, gt_line_width))
        panels.append(
            make_gt_query_channel_score_panel(attn_items, class_names,
                                              gt_query_channel_top_k,
                                              panel_size, font))
        panels.extend(
            make_gt_query_channel_panels(original, attn_items, meta,
                                         original_size, alpha, panel_size,
                                         font, gt_boxes, gt_line_width,
                                         class_names,
                                         gt_query_channel_top_k))

    if draw_feature_panels:
        panels.extend(
            make_feature_prepost_panels(original, attn_items, meta,
                                        original_size, alpha, panel_size,
                                        font, gt_boxes, gt_line_width,
                                        draw_feature_energy_panels))
        panels.append(make_frequency_panel(attn_items, panel_size, font))

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
    class_names = parse_class_names(args.class_names)
    feature_levels = parse_levels(args.feature_levels)
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
    patched_modules = patch_semantic_cross_attention(
        model, args.attn_reduce, feature_levels, args.freq_low_ratio,
        args.freq_high_ratio, args.feature_channel_top_k)
    if not patched_modules:
        raise RuntimeError('No SemanticCrossAttention modules were found.')

    rendered = []
    manifest_items = []
    for idx, image_path in enumerate(sampled, start=1):
        gt_boxes = lookup_gt_boxes(gt_index, image_path)
        with Image.open(image_path) as img:
            original_size = img.size
        clear_cached_attention(patched_modules)
        set_gt_query_context(patched_modules, original_size, gt_boxes)
        with torch.no_grad():
            result = inference_detector(model, str(image_path), test_pipeline)

        attn_items = collect_attention(model)
        if not attn_items:
            print(f'[{idx:03d}/{sample_n}] skipped, no attention: '
                  f'{image_path}')
            continue

        safe_stem = image_path.stem.replace(' ', '_')
        out_path = out_dir / f'{idx:03d}_{safe_stem}_scp_attn.png'
        level_manifests = []
        for item in attn_items:
            channel_stats = channel_gt_association(item, result.metainfo,
                                                   original_size, gt_boxes,
                                                   class_names)
            level_manifests.append({
                'name':
                item['name'],
                'level':
                item['level'],
                'patch_size':
                item['patch_size'],
                'attention_shape':
                item['attn_shape'],
                'attention_min':
                float(np.min(item['attn'])),
                'attention_max':
                float(np.max(item['attn'])),
                'attention_mean':
                float(np.mean(item['attn'])),
                'gt_query_attention': {
                    'query_pixel_count':
                    float(item.get('gt_query_count', 0.0)),
                    'attention_min':
                    float(np.min(item['gt_attn']))
                    if item.get('gt_attn') is not None else None,
                    'attention_max':
                    float(np.max(item['gt_attn']))
                    if item.get('gt_attn') is not None else None,
                    'attention_mean':
                    float(np.mean(item['gt_attn']))
                    if item.get('gt_attn') is not None else None,
                },
                'gt_query_channel_scores':
                [{
                    'channel':
                    int(ch),
                    'class_name':
                    class_label(int(ch), class_names),
                    'query_channel_score':
                    float(item['gt_query_class_scores'][ch]),
                    'key_channel_score':
                    float(item['gt_class_scores'][ch])
                    if item.get('gt_class_scores') is not None else None,
                } for ch in np.argsort(item['gt_query_class_scores'])[::-1]]
                if item.get('gt_query_class_scores') is not None else [],
                'feature_frequency':
                {
                    'top_changed_channels':
                    item['feature']['top_channels'],
                    'top_changed_channel_scores':
                    item['feature']['top_channel_scores'],
                    'pre':
                    item['feature']['pre_frequency'],
                    'post':
                    item['feature']['post_frequency'],
                    'change':
                    item['feature']['frequency_change'],
                    'freq_low_ratio':
                    item['feature']['freq_low_ratio'],
                    'freq_high_ratio':
                    item['feature']['freq_high_ratio'],
                } if item.get('feature') is not None else None,
                'class_attention':
                channel_stats,
                'top_channel_heatmaps': [
                    {
                        'channel':
                        int(ch),
                        'class_name':
                        class_label(int(ch), class_names),
                        'attention_score':
                        float(item['class_scores'][ch]),
                    }
                    for ch in np.argsort(item['class_scores'])[::-1]
                    [:max(int(args.channel_map_top_k), 0)]
                ],
            })

        render_visualization(image_path, attn_items, result.metainfo, out_path,
                             args.alpha, args.panel_size, gt_boxes,
                             args.gt_line_width, class_names,
                             args.class_top_k, args.channel_map_top_k,
                             not args.no_channel_heatmaps,
                             args.gt_query_channel_top_k,
                             not args.no_gt_query_panels,
                             not args.no_feature_panels,
                             not args.no_feature_energy_panels)
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
            'levels':
            level_manifests,
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
        'attn_reduce': args.attn_reduce,
        'alpha': args.alpha,
        'scp_class_names': class_names,
        'class_top_k': args.class_top_k,
        'channel_map_top_k': args.channel_map_top_k,
        'channel_heatmaps_drawn': not args.no_channel_heatmaps,
        'gt_query_channel_top_k': args.gt_query_channel_top_k,
        'gt_query_panels_drawn': not args.no_gt_query_panels,
        'feature_levels': feature_levels,
        'feature_panels_drawn': not args.no_feature_panels,
        'feature_channel_top_k': args.feature_channel_top_k,
        'feature_energy_panels_drawn': not args.no_feature_energy_panels,
        'freq_low_ratio': args.freq_low_ratio,
        'freq_high_ratio': args.freq_high_ratio,
        'gt_ann_file': str(gt_ann_file) if gt_ann_file else None,
        'gt_category_count': len(gt_categories),
        'gt_boxes_drawn': not args.no_gt_boxes and bool(gt_index),
        'contact_sheet': str(contact_sheet),
        'items': manifest_items,
    }
    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f'Wrote {len(rendered)} visualizations to {out_dir}')
    print(f'Contact sheet: {contact_sheet}')
    print(f'Manifest: {out_dir / "manifest.json"}')
    print('How to read: warm yellow/red regions are SCP semantic positions '
          'most attended by the HFP branch; GT boxes are drawn only on the '
          'Original panel and per-channel panels for target-location '
          'reference.')
    print('Class panel: each bar is sum_j mean_i(A_ij) * P(class|j); '
          'the value after / is the unweighted semantic presence baseline.')
    print('Channel heatmaps: each top channel panel shows '
          'mean_i(A_ij) * P(class|j), so the title channel id maps directly '
          'to the SCP semantic class name. Manifest fields inside_gt_mean, '
          'outside_gt_mean, gt_focus_ratio, and top_gt_categories help judge '
          'whether that semantic channel is associated with target boxes.')
    print('GT-query panels: GT-Q->SCP attn reduces A_ij using only query '
          'pixels inside GT boxes. GT query channel scores average '
          'sum_j A_ij * P(class|j) over those GT query pixels; the following '
          'GT-Q channel panels show the corresponding class-labeled spatial '
          'contribution heatmaps.')
    print('Feature panels: RMS panels aggregate all channels, while fmap chN '
          'panels are actual single-channel feature maps selected by the '
          'largest mean absolute SCP change. Frequency stats zero-center each '
          'channel, then report radial FFT low/high energy ratios; positive '
          'dH and dC mean the SCP fusion shifts the feature toward higher '
          'frequencies.')


if __name__ == '__main__':
    main()
