from typing import List

import torch
from torch import Tensor

from mmdet.models.dense_heads import ATSSHead
from mmdet.models.losses import (CIoULoss, DIoULoss, EIoULoss, GIoULoss,
                                 IoULoss, SIoULoss)
from mmdet.models.utils import images_to_levels, multi_apply
from mmdet.registry import MODELS
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean

# IoU-family losses return per-box scalar [num_pos]; per-coord losses
# (SmoothL1, L1) return [num_pos, 4]. Centerness weight broadcast must
# match the loss shape — we expand to [num_pos, 1] for per-coord losses.
_IOU_FAMILY = (IoULoss, GIoULoss, DIoULoss, CIoULoss, EIoULoss, SIoULoss)


@MODELS.register_module()
class CoATSSHead(ATSSHead):

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            centernesses: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor, ori_anchors, ori_labels,
         ori_bbox_targets) = cls_reg_targets

        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        losses_cls, losses_bbox, loss_centerness, \
            bbox_avg_factor = multi_apply(
                self.loss_by_feat_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                centernesses,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                avg_factor=avg_factor)

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))

        pos_coords = (ori_anchors, ori_labels, ori_bbox_targets, 'atss')
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_centerness=loss_centerness,
            pos_coords=pos_coords)

    def loss_by_feat_single(self, anchors: Tensor, cls_score: Tensor,
                            bbox_pred: Tensor, centerness: Tensor,
                            labels: Tensor, label_weights: Tensor,
                            bbox_targets: Tensor, avg_factor: float):
        """Override of ``ATSSHead.loss_by_feat_single`` to support
        ``reg_decoded_bbox=False`` with per-coordinate losses (SmoothL1, L1).

        Three changes vs the base implementation:

        1. When ``reg_decoded_bbox=False``, do NOT decode ``pos_bbox_pred``
           before the bbox loss — the loss must see raw delta predictions to
           match the encoded delta targets emitted by ``_get_targets_single``.
        2. ``centerness_target`` always needs decoded gt boxes (xyxy in image
           pixel space). When targets are encoded deltas, re-decode them via
           ``bbox_coder.decode`` before computing centerness.
        3. IoU-family losses return ``[num_pos]`` (broadcasts trivially with
           ``[num_pos]`` centerness weight). Per-coord losses return
           ``[num_pos, 4]`` and need the weight expanded to ``[num_pos, 1]``.
        """
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]

            # (Fix 2) centerness needs decoded gt boxes.
            if self.reg_decoded_bbox:
                pos_decoded_targets = pos_bbox_targets
            else:
                pos_decoded_targets = self.bbox_coder.decode(
                    pos_anchors, pos_bbox_targets)
            centerness_targets = self.centerness_target(
                pos_anchors, pos_decoded_targets)

            # (Fix 1) only decode preds when the loss is on decoded boxes.
            if self.reg_decoded_bbox:
                pred_for_loss = self.bbox_coder.decode(
                    pos_anchors, pos_bbox_pred)
                target_for_loss = pos_bbox_targets
            else:
                pred_for_loss = pos_bbox_pred
                target_for_loss = pos_bbox_targets

            # (Fix 3) per-coord losses need [num_pos, 1] weight.
            if isinstance(self.loss_bbox, _IOU_FAMILY):
                bbox_weight = centerness_targets
            else:
                bbox_weight = centerness_targets.unsqueeze(-1)

            loss_bbox = self.loss_bbox(
                pred_for_loss,
                target_for_loss,
                weight=bbox_weight,
                avg_factor=1.0)

            loss_centerness = self.loss_centerness(
                pos_centerness, centerness_targets, avg_factor=avg_factor)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()

    def get_targets(self,
                    anchor_list: List[List[Tensor]],
                    valid_flag_list: List[List[Tensor]],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    unmap_outputs: bool = True) -> tuple:
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list,
         sampling_results_list) = multi_apply(
             self._get_targets_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             batch_gt_instances,
             batch_img_metas,
             batch_gt_instances_ignore,
             unmap_outputs=unmap_outputs)
        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)

        ori_anchors = all_anchors
        ori_labels = all_labels
        ori_bbox_targets = all_bbox_targets
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, avg_factor, ori_anchors,
                ori_labels, ori_bbox_targets)
