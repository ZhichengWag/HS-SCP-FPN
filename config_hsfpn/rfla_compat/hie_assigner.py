from typing import Optional

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import get_box_tensor
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmdet.models.task_modules.assigners.base_assigner import BaseAssigner


@TASK_UTILS.register_module()
class HieAssigner(BaseAssigner):
    """RFLA hierarchical label assigner adapted to MMDetection 3.x."""

    def __init__(self,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=-1,
                 iou_calculator=dict(type='BboxDistanceMetric'),
                 assign_metric='kl',
                 topk=[2, 1],
                 ratio=1,
                 inside=False):
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.assign_metric = assign_metric
        self.topk = topk
        self.ratio = ratio
        self.inside = inside

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        priors = get_box_tensor(pred_instances.priors)
        gt_bboxes = get_box_tensor(gt_instances.bboxes)
        gt_labels = gt_instances.labels
        gt_bboxes_ignore = None
        if gt_instances_ignore is not None:
            gt_bboxes_ignore = get_box_tensor(gt_instances_ignore.bboxes)

        assign_on_cpu = (self.gpu_assign_thr > 0
                         and gt_bboxes.shape[0] > self.gpu_assign_thr)
        if assign_on_cpu:
            device = priors.device
            priors = priors.cpu()
            gt_bboxes = gt_bboxes.cpu()
            gt_labels = gt_labels.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()

        overlaps = self.iou_calculator(
            gt_bboxes, priors, mode=self.assign_metric)
        priors_rescaled = self.anchor_rescale(priors, self.ratio)
        overlaps_rescaled = self.iou_calculator(
            gt_bboxes, priors_rescaled, mode=self.assign_metric)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and priors.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    priors, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, priors, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assigned_gt_inds = self.assign_wrt_ranking(overlaps, self.topk[0],
                                                   gt_labels)
        assign_result = self.reassign_wrt_ranking(
            assigned_gt_inds, overlaps_rescaled, self.topk[1], gt_labels)

        if self.inside:
            assign_result.gt_inds *= self.inside_mask(priors, gt_bboxes,
                                                      assign_result.gt_inds)

        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_wrt_ranking(self, overlaps: Tensor, k: int,
                           gt_labels: Tensor) -> Tensor:
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return assigned_gt_inds

        max_overlaps, _ = overlaps.max(dim=0)
        k = min(k, num_bboxes)
        gt_max_overlaps, _ = overlaps.topk(k, dim=1, largest=True, sorted=True)
        assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < 0.8)] = 0

        for i in range(num_gts):
            for j in range(k):
                max_overlap_inds = overlaps[i, :] == gt_max_overlaps[i, j]
                assigned_gt_inds[max_overlap_inds] = i + 1
        return assigned_gt_inds

    def reassign_wrt_ranking(self, assign_result: Tensor, overlaps: Tensor,
                             k: int, gt_labels: Tensor) -> AssignResult:
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)
        mask_neg = assign_result <= 0
        mask_pos = assign_result > 0
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            assigned_labels = overlaps.new_full((num_bboxes, ),
                                                -1,
                                                dtype=torch.long)
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, max_overlaps,
                                assigned_labels)

        max_overlaps, _ = overlaps.max(dim=0)
        k = min(k, num_bboxes)
        gt_max_overlaps, _ = overlaps.topk(k, dim=1, largest=True, sorted=True)
        assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < 0.8)] = 0

        for i in range(num_gts):
            for j in range(k):
                max_overlap_inds = overlaps[i, :] == gt_max_overlaps[i, j]
                assigned_gt_inds[max_overlap_inds] = i + 1

        assigned_gt_inds = assigned_gt_inds * mask_neg + assign_result * mask_pos
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        pos_inds = torch.nonzero(
            assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] -
                                                  1]
        return AssignResult(num_gts, assigned_gt_inds, max_overlaps,
                            assigned_labels)

    @staticmethod
    def anchor_rescale(bboxes: Tensor, ratio: float) -> Tensor:
        bboxes = bboxes.clone()
        center_x = (bboxes[..., 2] + bboxes[..., 0]) / 2
        center_y = (bboxes[..., 3] + bboxes[..., 1]) / 2
        w = bboxes[..., 2] - bboxes[..., 0]
        h = bboxes[..., 3] - bboxes[..., 1]
        bboxes[..., 0] = center_x - w * ratio / 2
        bboxes[..., 1] = center_y - h * ratio / 2
        bboxes[..., 2] = center_x + w * ratio / 2
        bboxes[..., 3] = center_y + h * ratio / 2
        return bboxes

    @staticmethod
    def inside_mask(bboxes: Tensor, gt_bboxes: Tensor,
                    gt_inds: Tensor) -> Tensor:
        num_anchors = bboxes.size(0)
        num_gts = gt_bboxes.size(0)
        anchor_cx = (bboxes[..., 0] + bboxes[..., 2]) / 2
        anchor_cy = (bboxes[..., 1] + bboxes[..., 3]) / 2
        ext_gt_bboxes = gt_bboxes[:, None, :].expand(num_gts, num_anchors, 4)
        left = anchor_cx - ext_gt_bboxes[..., 0]
        right = ext_gt_bboxes[..., 2] - anchor_cx
        top = anchor_cy - ext_gt_bboxes[..., 1]
        bottom = ext_gt_bboxes[..., 3] - anchor_cy
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        inside_flag = bbox_targets.min(-1)[0] > 0
        length = range(gt_inds.size(0))
        return inside_flag[(gt_inds - 1).clamp(min=0), length]
