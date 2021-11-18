# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.modeling.make_layers import make_fc

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio)

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION)

        self.pooler = pooler
        self.head = head
        self.out_channels = head.out_channels

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor_channel_pool")
class FPN2MLPFeatureExtractor_channel_pool(nn.Module):
    """
    Heads for FPN for classification.
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor_channel_pool, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))
        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

    def forward(self, x, proposals=None, sup_features=None, oneStage=None):
        if proposals is not None:

            x = self.pooler(x, proposals)

            if oneStage:
                sup_feature_ROI = (torch.mul(sup_features[0], x) + torch.mul(sup_features[1], x) + torch.mul(sup_features[2], x) + torch.mul(sup_features[3], x) +
                                   torch.mul(sup_features[4], x) + torch.mul(sup_features[5], x) + torch.mul(sup_features[6], x) + torch.mul(sup_features[7], x) +
                                   torch.mul(sup_features[8], x) + torch.mul(sup_features[9], x) + torch.mul(sup_features[10], x) + torch.mul(sup_features[11], x) +
                                   torch.mul(sup_features[12], x) + torch.mul(sup_features[13], x) + torch.mul(sup_features[14], x))/int(len(sup_features))
            else:
                sup_feature_ROI = (torch.mul(sup_features[0], x) + torch.mul(sup_features[1], x) + torch.mul(sup_features[2], x) + torch.mul(sup_features[3], x) +
                                   torch.mul(sup_features[4], x) + torch.mul(sup_features[5], x) + torch.mul(sup_features[6], x) + torch.mul(sup_features[7], x) +
                                   torch.mul(sup_features[8], x) + torch.mul(sup_features[9], x) + torch.mul(sup_features[10], x) + torch.mul(sup_features[11], x) +
                                   torch.mul(sup_features[12], x) + torch.mul(sup_features[13], x) + torch.mul(sup_features[14], x) + torch.mul(sup_features[15], x) +
                                   torch.mul(sup_features[16], x) + torch.mul(sup_features[17], x) + torch.mul(sup_features[18], x) + torch.mul(sup_features[19], x)) / int(len(sup_features))


            sup_feature_ROI = sup_feature_ROI.view(sup_feature_ROI.size(0), -1)

            xr = F.relu(self.fc6r(sup_feature_ROI))
            xr = F.relu(self.fc7r(xr))  # [1024, 1024]

            xc = F.relu(self.fc6c(sup_feature_ROI))  # [1000, 1024]
            xc = F.relu(self.fc7c(xc))
            return xc, xr

        else:
            features = []
            for feature in x:
                feature = self.avgpooler(feature)
                feature = feature.view(feature.size(0), -1)
                feature = F.relu(self.fc6c(feature))
                feature = self.fc7c(feature)
                features.append(feature)
            return features

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification.
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))
        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

    def forward(self, x, proposals=None, sup_features=None, oneStage=None):
        if proposals is not None:
            roi_x = self.pooler(x, proposals)
            roi_x = roi_x.view(roi_x.size(0), -1)

            xr = F.relu(self.fc6r(roi_x))
            xr = F.relu(self.fc7r(xr))

            xc = F.relu(self.fc6c(roi_x))
            xc = F.relu(self.fc7c(xc))

            xc_cpe_normalized = None
            xc_sup_cpe_normalized = None

            return xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized

        else:
            features = []
            for feature in x:
                feature = self.avgpooler(feature)
                feature = feature.view(feature.size(0), -1)
                feature = F.relu(self.fc6c(feature))
                feature = self.fc7c(feature)
                features.append(feature)
            return features

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractorARRM")
class FPN2MLPFeatureExtractorARRM(nn.Module):
    """
    Heads for FPN for classification.
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractorARRM, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))
        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

        '''
        self.query_value = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1, dilation=1, stride=1)
        self.query_key   = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1, dilation=1, stride=1)

        self.sup_value   = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1, dilation=1, stride=1)
        self.sup_key     = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1, dilation=1, stride=1)

        self.catConv     = torch.nn.Conv2d(256*2, 256, kernel_size=3, padding=1, dilation=1, stride=1)

        self.bnn = nn.BatchNorm2d(256)
        '''

        self.query_value = torch.nn.Conv2d(256, 128, kernel_size=1)
        self.query_key   = torch.nn.Conv2d(256, 128, kernel_size=1)

        self.sup_value   = torch.nn.Conv2d(256, 128, kernel_size=1)
        self.sup_key     = torch.nn.Conv2d(256, 128, kernel_size=1)

        self.catConv     = torch.nn.Conv2d(256*2, 256, kernel_size=1)

        self.bnn = nn.BatchNorm2d(256)

    def forward(self, x, proposals=None, sup_features=None, oneStage=None):

        # sup_features： [15, 256, 12, 12]

        pooled_feat = self.pooler(x, proposals)

        sup_features = self.avgpooler(sup_features)  # [15, 256, 8, 8]
        batch_size = sup_features.size(0)  # 15

        support_mat = []

        if proposals is not None:

            for sup_feat in sup_features.chunk(batch_size, dim=0):

                # sup_feat [1, 256, 8, 8]

                sup_feat = sup_feat.repeat(pooled_feat.size(0), 1, 1, 1).unsqueeze(0)  # [1, 512, 256, 7, 7]
                support_mat += [sup_feat]

            support_mat = torch.cat(support_mat, 0)  # [15, 512, 256, 7, 7]

            B, C, H, W = pooled_feat.size()

            query_value  = self.query_value(pooled_feat)                                   # [B,   C/2, H, W]
            query_key    = self.query_key(pooled_feat).view(B, 128, -1).permute(0, 2, 1)   # [B, H*W, 256]

            final = []
            for i in range(batch_size):
                each_sup = support_mat[i]  # [512, 256, 7, 7]

                sup_value = self.sup_value(each_sup)               # [512, 128, 7, 7]
                sup_key = self.sup_key(each_sup).view(B, 128, -1)  # [512, 128, H*W]

                sim = torch.matmul(query_key, sup_key)
                sim = F.softmax(sim, dim=1)  # [512, H*W, H*W]

                sup_out = torch.matmul(sup_value.view(B, 128, -1), sim)

                sup_out = sup_out.view(B, 128, H, W)  # [512, 128, 7, 7]

                final_out = torch.cat((query_value, sup_out), dim=1)  # [512, 256, 7, 7]

                final_out = self.bnn(final_out)

                final.append(final_out)

            final = torch.stack(final, 0).mean(0)  # [512, 256, 7, 7]


            fuseFeature = self.catConv(torch.cat((pooled_feat, final), dim=1))          # [B, C, H, W]
            fuseFeature = fuseFeature.reshape(fuseFeature.size(0), -1)

            xr = F.relu(self.fc6r(fuseFeature))
            xr = F.relu(self.fc7r(xr))

            xc = F.relu(self.fc6c(fuseFeature))
            xc = F.relu(self.fc7c(xc))

            xc_cpe_normalized = None

            return xc, xr, xc_cpe_normalized

        else:
            features = []
            for feature in x:
                feature = self.avgpooler(feature)
                feature = feature.view(feature.size(0), -1)
                feature = F.relu(self.fc6c(feature))
                feature = self.fc7c(feature)
                features.append(feature)
            return features

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractorPSLoss")
class FPN2MLPFeatureExtractorPSLoss(nn.Module):
    """
    Heads for FPN for classification
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractorPSLoss, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))
        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

        self.head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
        )

    def forward(self, x, proposals=None, sup_features=None, oneStage=None):
        if proposals is not None:

            if oneStage:
                x = self.pooler(x, proposals)
                x = x.view(x.size(0), -1)

                xc = F.relu(self.fc6c(x))
                xc = F.relu(self.fc7c(xc))

                xr = F.relu(self.fc6r(x))
                xr = F.relu(self.fc7r(xr))

                xc_cpe_normalized     = None
                xc_sup_cpe_normalized = None

                return xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized

            else:
                x = self.pooler(x, proposals)
                x = x.view(x.size(0), -1)

                xc = F.relu(self.fc6c(x))
                xc = F.relu(self.fc7c(xc))

                xc_cpe = self.head(xc)  # [512, 1024]

                xc_cpe_normalized = F.normalize(xc_cpe, dim=1)

                xr = F.relu(self.fc6r(x))
                xr = F.relu(self.fc7r(xr))

                xc_sup_cpe_normalized = None

                return xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized
        else:
            features = []
            for feature in x:
                feature = self.avgpooler(feature)
                feature = feature.view(feature.size(0), -1)
                feature = F.relu(self.fc6c(feature))
                feature = self.fc7c(feature)
                features.append(feature)
            return features

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractorPCLoss")
class FPN2MLPFeatureExtractorPCLoss(nn.Module):
    """
    Heads for FPN for classification
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractorPCLoss, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))
        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

        self.fc_sup1 = make_fc(input_size, representation_size, use_gn)
        self.fc_sup2 = make_fc(representation_size, representation_size, use_gn)

        self.head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
        )

        self.head_sup = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
        )


    def forward(self, x, proposals=None, sup_features=None, oneStage=None):
        if proposals is not None:

            sup_features = F.interpolate(sup_features, size=(8, 8), mode='bilinear', align_corners=True)  # 下采样为和支持特征同样的尺度

            sup_features = sup_features.view(sup_features.size(0), -1)  # [15, 256*8*8]

            if oneStage:
                x = self.pooler(x, proposals)
                x = x.view(x.size(0), -1)

                xc = F.relu(self.fc6c(x))
                xc = F.relu(self.fc7c(xc))

                xr = F.relu(self.fc6r(x))
                xr = F.relu(self.fc7r(xr))

                xc_cpe_normalized = None
                xc_sup_cpe_normalized = None

                return xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized

            else:
                x = self.pooler(x, proposals)
                x = x.view(x.size(0), -1)
                xc = F.relu(self.fc6c(x))
                xc = F.relu(self.fc7c(xc))

                xc_cpe = self.head(xc)                                  # [512, 128]
                xc_cpe_normalized = F.normalize(xc_cpe, dim=1)          # [512, 128]

                xc_sup = F.relu(self.fc_sup1(sup_features))
                xc_sup = F.relu(self.fc_sup2(xc_sup))                   # [15, 1024]

                xc_sup_cpe = self.head_sup(xc_sup)                      # [15, 128]
                xc_sup_cpe_normalized = F.normalize(xc_sup_cpe, dim=1)  # [15, 128]

                xr = F.relu(self.fc6r(x))
                xr = F.relu(self.fc7r(xr))

                return xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized
        else:
            features = []
            for feature in x:
                feature = self.avgpooler(feature)
                feature = feature.view(feature.size(0), -1)
                feature = F.relu(self.fc6c(feature))
                feature = self.fc7c(feature)
                features.append(feature)
            return features

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractorPC_PSLoss")
class FPN2MLPFeatureExtractorPC_PSLoss(nn.Module):
    """
    Heads for FPN for classification
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractorPC_PSLoss, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))
        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

        self.fc_sup1 = make_fc(input_size, representation_size, use_gn)
        self.fc_sup2 = make_fc(representation_size, representation_size, use_gn)

        self.head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
        )

        self.head_sup = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
        )

    def forward(self, x, proposals=None, sup_features=None, oneStage=None):
        if proposals is not None:

            # print('sup_features: ', sup_features.shape)  # [15, 256, 8, 8]

            # sup_features = self.avgpooler(sup_features)  # [15, 256, 8, 8]

            sup_features = F.interpolate(sup_features, size=(8, 8), mode='bilinear', align_corners=True)  # 下采样为和支持特征同样的尺度

            sup_features = sup_features.view(sup_features.size(0), -1)  # [15, 256*8*8]

            if oneStage:
                x = self.pooler(x, proposals)
                x = x.view(x.size(0), -1)

                xc = F.relu(self.fc6c(x))
                xc = F.relu(self.fc7c(xc))

                xr = F.relu(self.fc6r(x))
                xr = F.relu(self.fc7r(xr))

                xc_cpe_normalized = None
                xc_sup_cpe_normalized = None

                return xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized
            else:
                x = self.pooler(x, proposals)
                x = x.view(x.size(0), -1)
                xc = F.relu(self.fc6c(x))
                xc = F.relu(self.fc7c(xc))

                xc_cpe = self.head(xc)                                  # [512, 128]
                xc_cpe_normalized = F.normalize(xc_cpe, dim=1)          # [512, 128]

                xc_sup = F.relu(self.fc_sup1(sup_features))
                xc_sup = F.relu(self.fc_sup2(xc_sup))                   # [15, 1024]

                xc_sup_cpe = self.head_sup(xc_sup)                      # [15, 128]
                xc_sup_cpe_normalized = F.normalize(xc_sup_cpe, dim=1)  # [15, 128]

                xr = F.relu(self.fc6r(x))
                xr = F.relu(self.fc7r(xr))

                return xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized
        else:
            features = []
            for feature in x:
                feature = self.avgpooler(feature)
                feature = feature.view(feature.size(0), -1)
                feature = F.relu(self.fc6c(feature))
                feature = self.fc7c(feature)
                features.append(feature)
            return features

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractorPrototypeLoss_meta")
class FPN2MLPFeatureExtractorPrototypeLoss_meta(nn.Module):
    """
    Heads for FPN for classification
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractorPrototypeLoss_meta, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))
        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

        self.fc_sup1 = make_fc(input_size, representation_size, use_gn)
        self.fc_sup2 = make_fc(representation_size, representation_size, use_gn)

        self.head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
        )

        self.head_sup = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
        )

        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))

        self.sup_conv1x1 = torch.nn.Conv2d(256, 256, kernel_size=1)
        self.sup_GAP = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid_sup = nn.Sigmoid()

    def forward(self, x, proposals=None, sup_features=None, oneStage=None):
        if proposals is not None:

            ncls = sup_features.shape[0]  # 15

            sup_features = F.interpolate(sup_features, size=(8, 8), mode='bilinear', align_corners=True)  # 下采样为和支持特征同样的尺度 [15, 256, 8, 8]
            sup_features = self.sup_conv1x1(sup_features)    # [15, 256, 8, 8]
            sup_GAP = self.sup_GAP(sup_features).view(ncls, -1)  # [15, 256]
            sup_attention = self.sigmoid_sup(sup_GAP).unsqueeze(-1).unsqueeze(-1)  # [15, 256, 1, 1]

            sup_features = sup_features.view(ncls, -1)  # [15, 256*8*8]

            if oneStage:
                x = self.pooler(x, proposals)   # [512, 256, 8, 8]

                roi_cls = x

                for i in range(ncls):
                    if (i == 0):
                        final_roi = roi_cls*(sup_attention[i].unsqueeze(0))
                    else:
                        final_roi += roi_cls*(sup_attention[i].unsqueeze(0))

                final_roi = final_roi.view(final_roi.size(0), -1)
                xc = F.relu(self.fc6c(final_roi))
                xc = F.relu(self.fc7c(xc))

                x = x.view(x.size(0), -1)
                xr = F.relu(self.fc6r(x))
                xr = F.relu(self.fc7r(xr))

                xc_cpe_normalized = None
                xc_sup_cpe_normalized = None

                return xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized

            else:
                x = self.pooler(x, proposals)
                roi_cls = x
                for i in range(ncls):
                    if (i == 0):
                        final_roi = roi_cls*(sup_attention[i].unsqueeze(0))
                    else:
                        final_roi += roi_cls*(sup_attention[i].unsqueeze(0))

                final_roi = final_roi.view(final_roi.size(0), -1)
                xc = F.relu(self.fc6c(final_roi))
                xc = F.relu(self.fc7c(xc))

                xc_cpe = self.head(xc)                                  # [512, 128]
                xc_cpe_normalized = F.normalize(xc_cpe, dim=1)          # [512, 128]


                xc_sup = F.relu(self.fc_sup1(sup_features))
                xc_sup = F.relu(self.fc_sup2(xc_sup))                   # [15, 1024]
                xc_sup_cpe = self.head_sup(xc_sup)                      # [15,  128]
                xc_sup_cpe_normalized = F.normalize(xc_sup_cpe, dim=1)  # [15,  128]

                x = x.view(x.size(0), -1)
                xr = F.relu(self.fc6r(x))
                xr = F.relu(self.fc7r(xr))



                return xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized
        else:
            features = []
            for feature in x:
                feature = self.avgpooler(feature)
                feature = feature.view(feature.size(0), -1)
                feature = F.relu(self.fc6c(feature))
                feature = self.fc7c(feature)
                features.append(feature)
            return features


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractorAffinityChannel")
class FPN2MLPFeatureExtractorAffinityChannel(nn.Module):
    """
    Heads for FPN for classification.
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractorAffinityChannel, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        # num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))
        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

        self.query_value = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1, dilation=1, stride=1)
        self.query_key = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1, dilation=1, stride=1)

        self.sup_value = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1, dilation=1, stride=1)
        self.sup_key = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1, dilation=1, stride=1)


    def forward(self, x, proposals=None, sup_features=None, oneStage=None):
        if proposals is not None:

            x = self.pooler(x, proposals)

            B, C, H, W = x.size()

            query_value = self.query_value(x)  # [B, C/2, H, W]
            query_key = self.query_key(x).reshape(-1, H*W)  # [BC/2, HW]

            sup_key = self.sup_key(sup_features).reshape(H*W, -1)  # [HW, NC/2]

            similarity = torch.mm(query_key, sup_key)   # [BC/2, NC/2]
            similarity = F.softmax(similarity, dim=-1)  # [BC/2, NC/2]

            sup_value = self.sup_value(sup_features).reshape(-1, H*W)  # [NC/2, HW]
            # print('sup_value: ', sup_value.shape)

            sup_value_similarity = torch.mm(similarity, sup_value).reshape(B, -1, H, W)  # [B, C/2, H, W]

            fuseFeature = torch.cat((query_value, sup_value_similarity), dim=1)  # [B, C, H, W]
            # print('fuseFeature: ', fuseFeature.shape)

            fuseFeature = fuseFeature.reshape(fuseFeature.size(0), -1)

            xr = F.relu(self.fc6r(fuseFeature))
            xr = F.relu(self.fc7r(xr))

            xc = F.relu(self.fc6c(fuseFeature))
            xc = F.relu(self.fc7c(xc))
            return xc, xr
        else:
            features = []
            for feature in x:
                feature = self.avgpooler(feature)
                feature = feature.view(feature.size(0), -1)
                feature = F.relu(self.fc6c(feature))
                feature = self.fc7c(feature)
                features.append(feature)
            return features

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractorAffinitySpatial")
class FPN2MLPFeatureExtractorAffinitySpatial(nn.Module):
    """
    Heads for FPN for classification.
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractorAffinitySpatial, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))
        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

        '''
        self.query_value = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1, dilation=1, stride=1)
        self.query_key   = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1, dilation=1, stride=1)

        self.sup_value   = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1, dilation=1, stride=1)
        self.sup_key     = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1, dilation=1, stride=1)
        '''
        self.query_value = torch.nn.Conv2d(256, 128, kernel_size=1)
        self.query_key = torch.nn.Conv2d(256, 128, kernel_size=1)

        self.sup_value = torch.nn.Conv2d(256, 128, kernel_size=1)
        self.sup_key = torch.nn.Conv2d(256, 128, kernel_size=1)


    def forward(self, x, proposals=None, sup_features=None, oneStage=None):
        if proposals is not None:

            x = self.pooler(x, proposals)

            B, C, H, W = x.size()

            query_value = self.query_value(x)  # [B, C/2, H, W]
            query_key = self.query_key(x).permute(0, 2, 3, 1).reshape(-1, 128)  # [BHW, C/2]

            sup_value = self.sup_value(sup_features).permute(0, 2, 3, 1).reshape(-1, 128)  # [NHW, C/2]
            sup_key = self.sup_key(sup_features).permute(1, 0, 2, 3).reshape(128, -1)      # [C/2, HWN]

            similarity = torch.mm(query_key, sup_key)   # [BHW, HWN]
            similarity = F.softmax(similarity, dim=-1)  # [BHW, HWN]

            sup_value_similarity = torch.mm(similarity, sup_value).reshape(B, H, W, 128).permute(0, 3, 1, 2)  # [B, C/2, H, W]

            fuseFeature = torch.cat((query_value, sup_value_similarity), dim=1)  # [B, C, H, W]

            fuseFeature = fuseFeature.reshape(fuseFeature.size(0), -1)

            xr = F.relu(self.fc6r(fuseFeature))
            xr = F.relu(self.fc7r(xr))

            xc = F.relu(self.fc6c(fuseFeature))
            xc = F.relu(self.fc7c(xc))
            return xc, xr

        else:
            features = []
            for feature in x:
                feature = self.avgpooler(feature)
                feature = feature.view(feature.size(0), -1)
                feature = F.relu(self.fc6c(feature))
                feature = self.fc7c(feature)
                features.append(feature)
            return features

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractorSpatialGAP_128")
class FPN2MLPFeatureExtractorSpatialGAP_128(nn.Module):
    """
    Heads for FPN for classification.
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractorSpatialGAP_128, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))
        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

        self.sup_GAP       = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.sup_GAP_key   = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1, dilation=1, stride=1)
        self.sup_GAP_value = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1, dilation=1, stride=1)

        self.query_value = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1, dilation=1, stride=1)
        self.query_key   = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1, dilation=1, stride=1)

        self.sup_value   = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1, dilation=1, stride=1)
        self.sup_key     = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1, dilation=1, stride=1)

        self.catConv     = torch.nn.Conv2d(128*3, 256, kernel_size=3, padding=1, dilation=1, stride=1)


    def forward(self, x, proposals=None, sup_features=None, oneStage=None):
        if proposals is not None:

            x = self.pooler(x, proposals)

            B, C, H, W = x.size()

            query_value  = self.query_value(x)                                              # [B,   C/2, H, W]
            query_key    = self.query_key(x).permute(0, 2, 3, 1).reshape(-1, 128)           # [BHW, C/2]

            sup_GAP       = self.sup_GAP(sup_features)                                       # [N,   C/2, 1, 1]
            sup_GAP_value = self.sup_GAP_key(sup_GAP).reshape(-1, 128)  # [N, C/2]
            sup_GAP_key   = self.sup_GAP_value(sup_GAP).permute(1, 0, 2, 3).reshape(128, -1)  # [C/2, N]


            sup_value = self.sup_value(sup_features).permute(0, 2, 3, 1).reshape(-1, 128)   # [NHW, C/2]
            sup_key = self.sup_key(sup_features).permute(1, 0, 2, 3).reshape(128, -1)       # [C/2, HWN]

            similarity = torch.mm(query_key, sup_key)           # [BHW, HWN]
            similarity = F.softmax(similarity, dim=-1)          # [BHW, HWN]

            similarity_chl = torch.mm(query_key, sup_GAP_key)  # [BHW, N]
            similarity_chl = F.softmax(similarity_chl, dim=-1)   # [BHW, N]

            sup_value_similarity     = torch.mm(similarity, sup_value).reshape(B, H, W, 128).permute(0, 3, 1, 2)      # [B, C/2, H, W]
            sup_value_similarity_chl = torch.mm(similarity_chl, sup_GAP_value).reshape(B, H, W, 128).permute(0, 3, 1, 2)  # [B, C/2, H, W]

            fuseFeature = F.relu(self.catConv(torch.cat((query_value, sup_value_similarity, sup_value_similarity_chl), dim=1)))             # [B, C, H, W]
            fuseFeature = fuseFeature.reshape(fuseFeature.size(0), -1)

            xr = F.relu(self.fc6r(fuseFeature))
            xr = F.relu(self.fc7r(xr))

            xc = F.relu(self.fc6c(fuseFeature))
            xc = F.relu(self.fc7c(xc))
            return xc, xr

        else:
            features = []
            for feature in x:
                feature = self.avgpooler(feature)
                feature = feature.view(feature.size(0), -1)
                feature = F.relu(self.fc6c(feature))
                feature = self.fc7c(feature)
                features.append(feature)
            return features

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractorSpatialGAP")
class FPN2MLPFeatureExtractorSpatialGAP(nn.Module):
    """
    Heads for FPN for classification.
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractorSpatialGAP, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))
        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

        self.sup_GAP       = torch.nn.AdaptiveAvgPool2d((1, 1))

        '''
        self.sup_GAP_key   = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, stride=1)
        self.sup_GAP_value = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, stride=1)

        self.query_value = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, stride=1)
        self.query_key   = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, stride=1)

        self.sup_value   = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, stride=1)
        self.sup_key     = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, stride=1)

        self.catConv     = torch.nn.Conv2d(256*3, 256, kernel_size=3, padding=1, dilation=1, stride=1)
        '''

        self.sup_GAP_key   = torch.nn.Conv2d(256, 256, kernel_size=1)
        self.sup_GAP_value = torch.nn.Conv2d(256, 256, kernel_size=1)

        self.query_value = torch.nn.Conv2d(256, 256, kernel_size=1)
        self.query_key   = torch.nn.Conv2d(256, 256, kernel_size=1)

        self.sup_value   = torch.nn.Conv2d(256, 256, kernel_size=1)
        self.sup_key     = torch.nn.Conv2d(256, 256, kernel_size=1)

        self.catConv     = torch.nn.Conv2d(256*3, 256, kernel_size=1)


    def forward(self, x, proposals=None, sup_features=None, oneStage=None):
        if proposals is not None:

            x = self.pooler(x, proposals)

            B, C, H, W = x.size()

            query_value  = self.query_value(x)                                              # [B,   C/2, H, W]
            query_key    = self.query_key(x).permute(0, 2, 3, 1).reshape(-1, 256)           # [BHW, C/2]

            sup_GAP       = self.sup_GAP(sup_features)                                       # [N,   C/2, 1, 1]
            sup_GAP_value = self.sup_GAP_key(sup_GAP).reshape(-1, 256)  # [N, C/2]
            sup_GAP_key   = self.sup_GAP_value(sup_GAP).permute(1, 0, 2, 3).reshape(256, -1)  # [C/2, N]


            sup_value = self.sup_value(sup_features).permute(0, 2, 3, 1).reshape(-1, 256)   # [NHW, C/2]
            sup_key = self.sup_key(sup_features).permute(1, 0, 2, 3).reshape(256, -1)       # [C/2, HWN]

            similarity = torch.mm(query_key, sup_key)            # [BHW, HWN]
            similarity = F.softmax(similarity, dim=-1)           # [BHW, HWN]

            similarity_chl = torch.mm(query_key, sup_GAP_key)    # [BHW, N]
            similarity_chl = F.softmax(similarity_chl, dim=-1)   # [BHW, N]

            sup_value_similarity     = torch.mm(similarity, sup_value).reshape(B, H, W, 256).permute(0, 3, 1, 2)      # [B, C/2, H, W]
            sup_value_similarity_chl = torch.mm(similarity_chl, sup_GAP_value).reshape(B, H, W, 256).permute(0, 3, 1, 2)  # [B, C/2, H, W]

            fuseFeature = F.relu(self.catConv(torch.cat((query_value, sup_value_similarity, sup_value_similarity_chl), dim=1)))             # [B, C, H, W]
            fuseFeature = fuseFeature.reshape(fuseFeature.size(0), -1)

            xr = F.relu(self.fc6r(fuseFeature))
            xr = F.relu(self.fc7r(xr))

            xc = F.relu(self.fc6c(fuseFeature))
            xc = F.relu(self.fc7c(xc))
            return xc, xr

        else:
            features = []
            for feature in x:
                feature = self.avgpooler(feature)
                feature = feature.view(feature.size(0), -1)
                feature = F.relu(self.fc6c(feature))
                feature = self.fc7c(feature)
                features.append(feature)
            return features

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractorSpatialMultiScale")
class FPN2MLPFeatureExtractorSpatialMultiScale(nn.Module):
    """
    Heads for FPN for classification.
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractorSpatialMultiScale, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))
        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

        self.sup_GAP       = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.sup_GAP2      = torch.nn.AdaptiveAvgPool2d((2, 2))
        self.sup_GAP4      = torch.nn.AdaptiveAvgPool2d((4, 4))

        self.sup_GAP_key   = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, stride=1)
        self.sup_GAP_value = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, stride=1)

        self.query_value = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, stride=1)
        self.query_key   = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, stride=1)

        self.sup_value   = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, stride=1)
        self.sup_key     = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, stride=1)

        self.catConv     = torch.nn.Conv2d(256*5, 256, kernel_size=3, padding=1, dilation=1, stride=1)

    def forward(self, x, proposals=None, sup_features=None, oneStage=None):
        if proposals is not None:

            x = self.pooler(x, proposals)

            B, C, H, W = x.size()

            query_value  = self.query_value(x)                                              # [B,   C, H, W]
            query_key    = self.query_key(x).permute(0, 2, 3, 1).reshape(-1, 256)           # [BHW, C]

            sup_GAP       = self.sup_GAP(sup_features)                                        # [N, C, 1, 1]
            sup_GAP_value = self.sup_GAP_key(sup_GAP).reshape(-1, 256)                        # [N, C]
            sup_GAP_key   = self.sup_GAP_value(sup_GAP).permute(1, 0, 2, 3).reshape(256, -1)  # [C, N]

            sup_GAP2       = self.sup_GAP2(sup_features)                                        # [N,     C, 2, 2]
            sup_GAP2_value = self.sup_GAP_key(sup_GAP2).reshape(-1, 256)                        # [N*2*2, C]
            sup_GAP2_key   = self.sup_GAP_value(sup_GAP2).permute(1, 0, 2, 3).reshape(256, -1)  # [C,     N*2*2]

            sup_GAP4       = self.sup_GAP4(sup_features)                                        # [N,     C, 4, 4]
            sup_GAP4_value = self.sup_GAP_key(sup_GAP4).reshape(-1, 256)                        # [N*4*4, C]
            sup_GAP4_key   = self.sup_GAP_value(sup_GAP4).permute(1, 0, 2, 3).reshape(256, -1)  # [C,     N*4*4]

            sup_value     = self.sup_value(sup_features).permute(0, 2, 3, 1).reshape(-1, 256)   # [NHW, C]
            sup_key       = self.sup_key(sup_features).permute(1, 0, 2, 3).reshape(256, -1)     # [C,   HWN]

            similarity = torch.mm(query_key, sup_key)             # [BHW, HWN]
            similarity = F.softmax(similarity, dim=-1)            # [BHW, HWN]

            similarity_chl = torch.mm(query_key, sup_GAP_key)     # [BHW, N]
            similarity_chl = F.softmax(similarity_chl, dim=-1)    # [BHW, N]

            similarity2_chl = torch.mm(query_key, sup_GAP2_key)   # [BHW, N*2*2]
            similarity2_chl = F.softmax(similarity2_chl, dim=-1)  # [BHW, N*2*2]

            similarity4_chl = torch.mm(query_key, sup_GAP4_key)   # [BHW, N*4*4]
            similarity4_chl = F.softmax(similarity4_chl, dim=-1)  # [BHW, N*4*4]

            sup_value_similarity     = torch.mm(similarity, sup_value).reshape(B, H, W, 256).permute(0, 3, 1, 2)             # [B, C, H, W]
            sup_value_similarity_chl = torch.mm(similarity_chl,   sup_GAP_value).reshape(B, H, W, 256).permute(0, 3, 1,  2)  # [B, C, H, W]
            sup_value_similarity2_chl = torch.mm(similarity2_chl, sup_GAP2_value).reshape(B, H, W, 256).permute(0, 3, 1, 2)  # [B, C, H, W]
            sup_value_similarity4_chl = torch.mm(similarity4_chl, sup_GAP4_value).reshape(B, H, W, 256).permute(0, 3, 1, 2)  # [B, C, H, W]

            fuseFeature = F.relu(self.catConv(torch.cat((query_value, sup_value_similarity, sup_value_similarity_chl,
                                                         sup_value_similarity2_chl, sup_value_similarity4_chl), dim=1)))             # [B, C, H, W]
            fuseFeature = fuseFeature.reshape(fuseFeature.size(0), -1)

            xr = F.relu(self.fc6r(fuseFeature))
            xr = F.relu(self.fc7r(xr))

            xc = F.relu(self.fc6c(fuseFeature))
            xc = F.relu(self.fc7c(xc))
            return xc, xr

        else:
            features = []
            for feature in x:
                feature = self.avgpooler(feature)
                feature = feature.view(feature.size(0), -1)
                feature = F.relu(self.fc6c(feature))
                feature = self.fc7c(feature)
                features.append(feature)
            return features

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractorAffinitySpatialChannel")
class FPN2MLPFeatureExtractorAffinitySpatialChannel(nn.Module):
    """
    Heads for FPN for classification.
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractorAffinitySpatialChannel, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        # num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))
        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

        self.query_value = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1, dilation=1, stride=1)
        self.query_key = torch.nn.Conv2d(256, 64, kernel_size=3, padding=1, dilation=1, stride=1)

        self.sup_value = torch.nn.Conv2d(256, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        self.sup_key = torch.nn.Conv2d(256, 64, kernel_size=3, padding=1, dilation=1, stride=1)

    def forward(self, x, proposals=None, sup_features=None, oneStage=None):
        if proposals is not None:

            x = self.pooler(x, proposals)

            B, C, H, W = x.size()

            query_value= self.query_value(x)                              # [B,    C/2, H, W]
            query_key_C  = self.query_key(x).reshape(-1, H*W)             # [BC/4, HW]
            sup_key_C    = self.sup_key(sup_features).reshape(H * W, -1)  # [HW,   NC/4]
            similarity_C = torch.mm(query_key_C, sup_key_C)               # [BC/4, NC/4]
            similarity_C = F.softmax(similarity_C, dim=-1)                # [BC/4, NC/4]

            query_key_S  = self.query_key(x).permute(0, 2, 3, 1).reshape(-1, 64)           # [BHW, C/4]
            sup_key_S    = self.sup_key(sup_features).permute(1, 0, 2, 3).reshape(64, -1)  # [C/4, HWN]
            similarity_S = torch.mm(query_key_S, sup_key_S)                                # [BHW, HWN]
            similarity_S = F.softmax(similarity_S, dim=-1)                                 # [BHW, HWN]

            sup_value_C = self.sup_value(sup_features).reshape(-1, H*W)                     # [NC/4, HW]
            sup_value_S = self.sup_value(sup_features).permute(0, 2, 3, 1).reshape(-1, 64)  # [NHW, C/4]

            sup_value_similarity_S = torch.mm(similarity_S, sup_value_S).reshape(B, -1, H, W)                       # [B, C/4, H, W]

            sup_value_similarity_C = torch.mm(similarity_C, sup_value_C).reshape(B, H, W, -1).permute(0, 3, 1, 2)  # [B, C/4, H, W]


            fuseFeature = torch.cat((sup_value_similarity_S, sup_value_similarity_C), dim=1)  # [B, C/2, H, W]
            fuseFeature = torch.cat((query_value, fuseFeature), dim=1)                        # [B, C, H, W]

            # print('fuseFeature: ', fuseFeature.shape)

            fuseFeature = fuseFeature.view(fuseFeature.size(0), -1)

            xr = F.relu(self.fc6r(fuseFeature))
            xr = F.relu(self.fc7r(xr))  # [1024, 1024]

            xc = F.relu(self.fc6c(fuseFeature))  # [1000, 1024]
            xc = F.relu(self.fc7c(xc))
            return xc, xr

        else:
            features = []
            for feature in x:
                feature = self.avgpooler(feature)
                feature = feature.view(feature.size(0), -1)
                feature = F.relu(self.fc6c(feature))
                feature = self.fc7c(feature)
                features.append(feature)
            return features

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractorContext")
class FPN2MLPFeatureExtractorContext(nn.Module):
    """
    Heads for FPN for classification
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractorContext, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO

        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))
        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)

        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

        self.avgpool1x1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool1x1_Conv = nn.Conv2d(in_channels, in_channels, 1, 1, bias=False)
        self.avgpool1x1_Rule = nn.ReLU(inplace=True)

        self.avgpool2x2 = nn.AdaptiveAvgPool2d((2, 2))
        self.avgpool2x2_Conv = nn.Conv2d(in_channels, in_channels, 1, 1, bias=False)
        self.avgpool2x2_Rule = nn.ReLU(inplace=True)

        self.avgpool3x3 = nn.AdaptiveAvgPool2d((3, 3))
        self.avgpool3x3_Conv = nn.Conv2d(in_channels, in_channels, 1, 1, bias=False)
        self.avgpool3x3_Rule = nn.ReLU(inplace=True)

        self.avgpool6x6 = nn.AdaptiveAvgPool2d((6, 6))
        self.avgpool6x6_Conv = nn.Conv2d(in_channels, in_channels, 1, 1, bias=False)
        self.avgpool6x6_Rule = nn.ReLU(inplace=True)

        self.ConvCat = nn.Conv2d(4*256, 256, 1, 1, bias=False)

    def forward(self, x, proposals=None):
        if proposals is not None:

            x = self.pooler(x, proposals)  # [512, 256, 8, 8] # 感兴趣对齐之后的feature maps

            # x_supCat: [class, 256, 8, 8]
            B = x.shape[0]
            C = x.shape[1]
            H = x.shape[2]
            W = x.shape[3]

            x_t = x.view(B, -1, H * W).permute(0, 2, 1)                       # [B, 8*8, C]

            avgpool1x1 = self.avgpool1x1(x).view(B, -1, 1*1)                  # [B, C,   1*1]
            scale1 = torch.bmm(x_t, avgpool1x1)                               # [B, 8*8, 1*1]
            scale1 = F.softmax(scale1, dim=-1).permute(0, 2, 1)               # [B, 1*1, 8*8]
            scale1 = torch.bmm(avgpool1x1, scale1).view(B, C, H, W)           # [B, C,   8, 8]
            scale1 = self.avgpool1x1_Rule(self.avgpool1x1_Conv(scale1))

            avgpool2x2 = self.avgpool2x2(x).view(B, -1, 2*2)                  # [B, C,   2*2]
            scale2 = torch.bmm(x_t, avgpool2x2)                               # [B, 8*8, 2*2]
            scale2 = F.softmax(scale2, dim=-1).permute(0, 2, 1)               # [B, 2*2, 8*8]
            scale2 = torch.bmm(avgpool2x2, scale2).view(B, C, H, W)           # [B, C,   8, 8]
            scale2 = self.avgpool2x2_Rule(self.avgpool2x2_Conv(scale2))

            avgpool3x3 = self.avgpool3x3(x).view(B, -1, 3*3)                  # [B, C,   3*3]
            scale3 = torch.bmm(x_t, avgpool3x3)                               # [B, 8*8, 3*3]
            scale3 = F.softmax(scale3, dim=-1).permute(0, 2, 1)               # [B, 3*3, 8*8]
            scale3 = torch.bmm(avgpool3x3, scale3).view(B, C, H, W)           # [B, C,   8, 8]
            scale3 = self.avgpool3x3_Rule(self.avgpool3x3_Conv(scale3))

            avgpool6x6 = self.avgpool6x6(x).view(B, -1, 6*6)                  # [B, C,   6*6]
            scale6 = torch.bmm(x_t, avgpool6x6)                               # [B, 8*8, 6*6]
            scale6 = F.softmax(scale6, dim=-1).permute(0, 2, 1)               # [B, 6*6, 8*8]
            scale6 = torch.bmm(avgpool6x6, scale6).view(B, C, H, W)           # [B, C,   8, 8]
            scale6 = self.avgpool6x6_Rule(self.avgpool6x6_Conv(scale6))

            x = x + self.ConvCat(torch.cat([scale1, scale2, scale3, scale6], dim=1))

            x = x.view(x.size(0), -1)
            xc = F.relu(self.fc6c(x))
            xc = self.fc7c(xc)
            xr = F.relu(self.fc6r(x))
            xr = F.relu(self.fc7r(xr))
            return xc, xr
        else:
            features = []
            for feature in x:
                feature = self.avgpooler(feature)
                feature = feature.view(feature.size(0), -1)
                feature = F.relu(self.fc6c(feature))
                feature = self.fc7c(feature)
                features.append(feature)
            return features

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPNXconv1fcFeatureExtractor")
class FPNXconv1fcFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification.
    """

    def __init__(self, cfg, in_channels):
        super(FPNXconv1fcFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        conv_head_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM
        num_stacked_convs = cfg.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.MODEL.ROI_BOX_HEAD.DILATION

        xconvs = []
        for ix in range(num_stacked_convs):
            xconvs.append(
                nn.Conv2d(
                    in_channels,
                    conv_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=False if use_gn else True
                )
            )
            in_channels = conv_head_dim
            if use_gn:
                xconvs.append(group_norm(in_channels))
            xconvs.append(nn.ReLU(inplace=True))

        self.add_module("xconvs", nn.Sequential(*xconvs))
        for modules in [self.xconvs,]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if not use_gn:
                        torch.nn.init.constant_(l.bias, 0)

        input_size = conv_head_dim * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.fc6 = make_fc(input_size, representation_size, use_gn=False)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.xconvs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x

def make_roi_box_feature_extractor(cfg, in_channels):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR]
    return func(cfg, in_channels)