import torch

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import get_box_tensor


@TASK_UTILS.register_module()
class BboxDistanceMetric:
    """RFLA bbox distance metrics for MMDetection 3.x task utils."""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        bboxes1 = get_box_tensor(bboxes1)
        bboxes2 = get_box_tensor(bboxes2)
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)


def bbox_overlaps(bboxes1,
                  bboxes2,
                  mode='iou',
                  is_aligned=False,
                  eps=1e-6):
    assert mode in [
        'iou', 'iof', 'giou', 'wd', 'kl', 'center_distance2', 'exp_kl',
        'kl_10'
    ], f'Unsupported mode {mode}'
    assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]

    batch_shape = bboxes1.shape[:-2]
    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        return bboxes1.new(batch_shape + (rows, cols))

    if is_aligned:
        raise NotImplementedError('RFLA distance metric uses unaligned boxes.')

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
    rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[..., 0] * wh[..., 1]

    if mode == 'iof':
        union = area1[..., None].clamp(min=eps)
    else:
        union = area1[..., None] + area2[..., None, :] - overlap
        union = union.clamp(min=eps)
    ious = overlap / union

    if mode in ['iou', 'iof']:
        return ious

    if mode == 'giou':
        enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                bboxes2[..., None, :, :2])
        enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                bboxes2[..., None, :, 2:])
        enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
        enclose_area = (enclose_wh[..., 0] * enclose_wh[..., 1]).clamp(
            min=eps)
        return ious - (enclose_area - union) / enclose_area

    center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
    center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
    center_delta = center1 - center2

    if mode == 'center_distance2':
        return (center_delta[..., 0] * center_delta[..., 0] +
                center_delta[..., 1] * center_delta[..., 1] + eps)

    w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
    h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
    w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
    h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

    if mode in ['kl', 'kl_10', 'exp_kl']:
        kl = (w2**2 / w1**2 + h2**2 / h1**2 +
              4 * center_delta[..., 0]**2 / w1**2 +
              4 * center_delta[..., 1]**2 / h1**2 +
              torch.log(w1**2 / w2**2) + torch.log(h1**2 / h2**2) - 2) / 2
        if mode == 'kl':
            return 1 / (1 + kl)
        if mode == 'kl_10':
            return 1 / (10 + kl)
        return torch.exp(-kl / 10)

    center_distance = (center_delta[..., 0] * center_delta[..., 0] +
                       center_delta[..., 1] * center_delta[..., 1] + eps)
    wh_distance = ((w1 - w2)**2 + (h1 - h2)**2) / 4
    return 1 / (1 + center_distance + wh_distance)
