# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F
from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator

class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """
    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
    def forward(self, features, proposals, targets=None, sup_features=None, supTarget=None, oneStage=None):

        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets

        Returns:
            x (Tensor): the result of the feature extractor.
            proposals (list[BoxList]): during training, the subsampled proposals are returned. During testing, the predicted boxlists are returned.
            losses (dict[Tensor]): During training, returns the losses for the head. During testing, returns an empty dict.
        """

        if self.training:
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized = self.feature_extractor(features, proposals, sup_features, oneStage)  # [20, 128]

        # xc_cpe_normalized = [1000, 128]

        class_logits, box_regression = self.predictor(xc, xr) # class_logits.shape:  torch.Size([512, 16]); box_regression.shape:  torch.Size([512, 64])


        '''
        class_probxdx = F.softmax(class_logits, -1)  # [1000, 21]
        Nums = class_probxdx.size(0)
        index_proposal = []
        index_max_row  = []
        index_max      = []
        for xx in range(Nums):
            each_class_prob = class_probxdx[xx]  # [21]
            ff = torch.argmax(each_class_prob)
            ff2 = torch.max(each_class_prob)
            if ff > 0:
                index_proposal.append(xx)
                index_max_row.append(ff)  # mei yi hang de zui da zhi de index
                index_max.append(ff2)     # mei yi hang de zui da zhi
        if index_max == []:
            final_logit = class_logits[0]  # [21]
        else:
            ff3 = max(index_max)
            final_index_proposal = index_proposal[index_max.index(ff3)]
            final_logit = class_logits[final_index_proposal]  # [21]
        '''


        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return xc, result, {}

        if xc_cpe_normalized is None:
            loss_classifier, loss_box_reg = self.loss_evaluator([class_logits], [box_regression], xc_cpe_normalized, xc_sup_cpe_normalized, supTarget)
            return (xc, proposals, dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg))
        else:
            loss_classifier, loss_box_reg, loss_pc = self.loss_evaluator([class_logits], [box_regression], xc_cpe_normalized, xc_sup_cpe_normalized, supTarget)
            return (xc, proposals, dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_pc=loss_pc))

def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class and make it a parameter in the config.
    """
    return ROIBoxHead(cfg, in_channels)