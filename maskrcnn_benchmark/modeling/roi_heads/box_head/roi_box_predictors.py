# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry

from aug import ISDAaug
from augReg import ISDAaugReg
import torch
from torch import nn
from torch.nn import functional as F

@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred

@registry.ROI_BOX_PREDICTOR.register("FPNPredictorISDA")
class FPNPredictorISDA(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictorISDA, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)
        self.num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        # self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)
        self.bbox_pred = nn.Linear(representation_size, num_classes)
        # self.bbox_pred = nn.Linear(representation_size, 4)

        self.ISDAaug = ISDAaug(1024, num_classes)
        # self.ISDAaugReg = ISDAaugReg(1024, num_classes)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, xc, xr=None, label=None):
        if xr is not None:
            scores = self.cls_score(xc)  # [512, 16]
            output = self.ISDAaug(self.cls_score, scores, xc, label, 5)  # [512, 16]
            # bbox_deltas = self.bbox_pred(xr).repeat(1, self.num_bbox_reg_classes)  # [512, C*4]
            bbox_deltas = self.bbox_pred(xr).repeat(1, 4)
            # outputReg = self.ISDAaugReg(self.bbox_pred, bbox_deltas, xr, label, 5)
            # print('bbox_deltas: ', bbox_deltas.shape)
            return output, bbox_deltas
        else:
            xcs = []
            for feature in xc:
                xcs.append(self.cls_score(feature))
            return xcs

@registry.ROI_BOX_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)
        self.num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        # self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)
        self.bbox_pred = nn.Linear(representation_size, 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, xc, xr=None):
        if xr is not None:
            scores = self.cls_score(xc)
            bbox_deltas = self.bbox_pred(xr).repeat(1, self.num_bbox_reg_classes)
            return scores, bbox_deltas
        else:
            xcs = []
            for feature in xc:
                xcs.append(self.cls_score(feature))
            return xcs

@registry.ROI_BOX_PREDICTOR.register("FPNPredictorLearn2Compare")
class FPNPredictorLearn2Compare(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictorLearn2Compare, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels
        # self.cls_score = nn.Linear(representation_size, num_classes)
        self.num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, 4)

        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, f_roi_logits, sup_features_logits=None, xr=None, label=None):
        f_roi_reshape_x = f_roi_logits.unsqueeze(0)  # [1, 512, 1024]
        f_roi_reshape_x = F.normalize(f_roi_reshape_x, dim=-1)  # [1, 512, 1024]

        sup_batch = sup_features_logits.shape[0]

        sup_features_reshpe_x = sup_features_logits.unsqueeze(0)  # [1, 16, 1024]
        sup_features_reshpe_x = F.normalize(sup_features_reshpe_x, dim=-1)  # [1, 16, 1024]

        cls_score = torch.bmm(f_roi_reshape_x, sup_features_reshpe_x.permute(0, 2, 1))  # cos_similarity [1, 512, 16]
        cls_score = 20 * cls_score
        cls_score = cls_score.view(-1, sup_batch)

        bbox_deltas = self.bbox_pred(xr).repeat(1, self.num_bbox_reg_classes)

        return cls_score, bbox_deltas


@registry.ROI_BOX_PREDICTOR.register("FPNPredictorCosine")
class FPNPredictorCosine(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictorCosine, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        # self.cls_score = nn.Linear(representation_size, num_classes)
        self.num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, 4)

        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, f_roi_logits, sup_features_logits=None, xr=None):

        f_roi_reshape_x = f_roi_logits                          # [512, 256]
        f_roi_reshape_x = F.normalize(f_roi_reshape_x, dim=-1)  # [512, 256]

        sup_features_reshpe_x = sup_features_logits                         # [16, 256]
        sup_features_reshpe_x = F.normalize(sup_features_reshpe_x, dim=-1)  # [16, 256]

        cls_score = torch.mm(f_roi_reshape_x, sup_features_reshpe_x.t())  # cos_similarity [512, 16]
        cls_score = 0.3*cls_score

        bbox_deltas = self.bbox_pred(xr).repeat(1, self.num_bbox_reg_classes)

        return cls_score, bbox_deltas


@registry.ROI_BOX_PREDICTOR.register("FPNCosinePredictor")
class FPNCosinePredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNCosinePredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)
        self.num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        # self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)
        self.bbox_pred = nn.Linear(representation_size, 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, xc, xr=None):
        # if x.ndimension() == 4:
        #    assert list(x.shape[2:]) == [1, 1]
        #    x = x.view(x.size(0), -1)
        if xr is not None:

            xc_norm = torch.norm(xc, p=2, dim=1).unsqueeze(1).expand_as(xc)
            xc_normalized = xc.div(xc_norm + 1e-5)

            temp_norm = torch.norm(self.cls_score.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.cls_score.weight.data)
            self.cls_score.weight.data = self.cls_score.weight.data.div(temp_norm + 1e-5)
            cos_dist = self.cls_score(xc_normalized)

            scores = 20.0 * cos_dist
            bbox_deltas = self.bbox_pred(xr).repeat(1, self.num_bbox_reg_classes)
            return scores, bbox_deltas
        else:
            xcs = []
            for feature in xc:
                xc_norm = torch.norm(feature, p=2, dim=1).unsqueeze(1).expand_as(feature)
                xc_normalized = xc.div(xc_norm + 1e-5)
                temp_norm = torch.norm(self.cls_score.weight.data, p=2, dim=1).unsqueeze(1).expand_as(
                    self.cls_score.weight.data)
                self.cls_score.weight.data = self.cls_score.weight.data.div(temp_norm + 1e-5)
                cos_dist = self.cls_score(xc_normalized)
                scores = self.scale * cos_dist
                xcs.append(scores)
            return xcs


def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)