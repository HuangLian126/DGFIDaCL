# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Implements the Generalized R-CNN framework.
"""

import torch
from torch import nn
from maskrcnn_benchmark.structures.image_list import to_image_list
from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads

# from DaFI import DAFI
# from DaFIv2 import DAFIV2
# from DaFIv3 import DAFIV3
from DGFI import DAFIV4

# from DaFIv6 import DAFIV6
# from DRD import DenseRelationDistill
# from arrm import ARRM

class GeneralizedRCNN(nn.Module):

    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        # self.DAFIV6_opt = DAFIV6(256, 256, 256, dense_sum=True)
        self.DAFIV4_opt = DAFIV4(256, 32, 128, dense_sum=True)

        # self.conv_before = nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1)

    def forward(self, images, targets=None, sup=None, supTarget=None, oneStage=None, gamma=None, margin=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if sup is not None:

            Batch = to_image_list(sup).tensors.shape[0]
            H = to_image_list(sup).tensors.shape[3]
            W = to_image_list(sup).tensors.shape[4]
            sup_imgs = to_image_list(sup).tensors.reshape(Batch, -1, H, W)  # [15, 3, 192, 192]
            sup_feats = self.backbone(sup_imgs)[2]

        images = to_image_list(images)
        features = self.backbone(images.tensors)

        '''
        before_features = []
        for i in range(len(features)):
            before_features.append(self.conv_before(features[i]))
        before_features = tuple(before_features)
        '''

        features = self.DAFIV4_opt(features, sup_feats)

        proposals, proposal_losses = self.rpn(images, features, targets)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets, sup_feats, supTarget, oneStage)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result