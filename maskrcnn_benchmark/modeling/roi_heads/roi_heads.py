# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head

class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single head.
    """
    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None, sup_features=None, supTarget=None, oneStage=None):
        losses = {}

        x, detections, loss_box = self.box(features, proposals, targets, sup_features, supTarget, oneStage)

        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            if (self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR):
                mask_features = x
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)
        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features

            if (self.training and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR):
                keypoint_features = x

            x, detections, loss_keypoint = self.keypoint(keypoint_features, detections, targets)
            losses.update(loss_keypoint)
        return x, detections, losses

def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together afterwards.
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))

    # combine individual heads in a single module.
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads