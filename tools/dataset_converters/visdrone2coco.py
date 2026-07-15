# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm


CLASSES = (
    'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle',
    'awning-tricycle', 'bus', 'motor')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert VisDrone2019-DET annotations to COCO format')
    parser.add_argument('data_root', type=Path)
    parser.add_argument(
        '--splits', nargs='+', default=['train', 'val'], choices=['train', 'val'])
    parser.add_argument('--output-dir', type=Path, default=None)
    return parser.parse_args()


def convert_split(data_root, output_dir, split):
    split_dir = data_root / f'VisDrone2019-DET-{split}'
    image_dir = split_dir / 'images'
    annotation_dir = split_dir / 'annotations'
    image_paths = sorted(image_dir.glob('*.jpg'))
    if not image_paths:
        raise FileNotFoundError(f'No JPG images found in {image_dir}')

    output = {
        'images': [],
        'annotations': [],
        'categories': [
            {'id': category_id, 'name': name}
            for category_id, name in enumerate(CLASSES, start=1)
        ],
    }
    annotation_id = 1
    skipped = 0

    for image_id, image_path in enumerate(tqdm(image_paths, desc=split), start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        output['images'].append({
            'id': image_id,
            'file_name': image_path.name,
            'width': width,
            'height': height,
        })

        annotation_path = annotation_dir / f'{image_path.stem}.txt'
        if not annotation_path.is_file():
            raise FileNotFoundError(f'Missing annotation: {annotation_path}')
        for line_number, line in enumerate(
                annotation_path.read_text(encoding='utf-8-sig').splitlines(), start=1):
            if not line.strip():
                continue
            fields = line.rstrip(',').split(',')
            if len(fields) != 8:
                raise ValueError(
                    f'{annotation_path}:{line_number}: expected 8 fields')
            x, y, box_width, box_height, score, category_id, truncation, occlusion = (
                int(value) for value in fields)
            if score == 0 or category_id not in range(1, len(CLASSES) + 1):
                skipped += 1
                continue
            if box_width <= 0 or box_height <= 0:
                skipped += 1
                continue

            output['annotations'].append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': category_id,
                'bbox': [x, y, box_width, box_height],
                'area': box_width * box_height,
                'iscrowd': 0,
                'truncation': truncation,
                'occlusion': occlusion,
            })
            annotation_id += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'visdrone2019_det_{split}_coco.json'
    output_path.write_text(json.dumps(output), encoding='utf-8')
    print(f'{output_path}: {len(output["images"])} images, '
          f'{len(output["annotations"])} annotations, {skipped} skipped')


def main():
    args = parse_args()
    output_dir = args.output_dir or args.data_root / 'annotations'
    for split in args.splits:
        convert_split(args.data_root, output_dir, split)


if __name__ == '__main__':
    main()
