import logging

import cv2
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from mmengine.logging import print_log


def _apply_detr_amp_patch():
    """Cast bbox_overlaps result to scores.dtype in DETRHead.loss_by_feat_single
    to avoid an fp16/fp32 mismatch under QualityFocalLoss (Co-DINO path)."""
    import torch
    from mmdet.models.dense_heads import detr_head as _detr_head_mod
    from mmdet.models.losses import QualityFocalLoss
    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
    from mmdet.utils import reduce_mean

    _orig = _detr_head_mod.DETRHead.loss_by_feat_single

    def _patched(self, cls_scores, bbox_preds, batch_gt_instances,
                 batch_img_metas):
        if not isinstance(self.loss_cls, QualityFocalLoss):
            return _orig(self, cls_scores, bbox_preds, batch_gt_instances,
                         batch_img_metas)

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           batch_gt_instances, batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = (num_total_pos * 1.0
                          + num_total_neg * self.bg_cls_weight)
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        scores = label_weights.new_zeros(labels.shape)
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
        pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
        pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
        iou = bbox_overlaps(pos_decode_bbox_pred.detach(),
                            pos_decode_bbox_targets, is_aligned=True)
        scores[pos_inds] = iou.to(scores.dtype)
        loss_cls = self.loss_cls(
            cls_scores, (labels, scores), label_weights,
            avg_factor=cls_avg_factor)

        num_total_pos_t = loss_cls.new_tensor([num_total_pos])
        num_total_pos_t = torch.clamp(reduce_mean(num_total_pos_t),
                                      min=1).item()

        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor(
                [img_w, img_h, img_w, img_h]
            ).unsqueeze(0).repeat(bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos_t)
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights,
            avg_factor=num_total_pos_t)
        return loss_cls, loss_bbox, loss_iou

    _detr_head_mod.DETRHead.loss_by_feat_single = _patched


try:
    _apply_detr_amp_patch()
except Exception as _e:  # noqa: BLE001
    print_log(f'[custom_transforms] DETR AMP patch skipped: {_e}',
              logger='current', level=logging.WARNING)


@TRANSFORMS.register_module()
class CLAHE(BaseTransform):
    """Per-channel Contrast Limited Adaptive Histogram Equalization."""

    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tuple(tile_grid_size)

    def transform(self, results):
        img = results['img']
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        if img.ndim == 3:
            for c in range(img.shape[2]):
                img[:, :, c] = clahe.apply(img[:, :, c])
        else:
            img = clahe.apply(img)
        results['img'] = img
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'clip_limit={self.clip_limit}, '
                f'tile_grid_size={self.tile_grid_size})')
