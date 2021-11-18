import torch
import torch.nn.functional as F
import torch.nn as nn

class DFGI(nn.Module):

    def __init__(self, indim, keydim, valdim, dense_sum=False):
        super(DFGI, self).__init__()

        self.sum = dense_sum

        self.sup_GAP_key   = torch.nn.Conv2d(indim, keydim, kernel_size=1)

        self.sup_value = torch.nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.sup_key = torch.nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.query_value0   = torch.nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.query_key0     = torch.nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.query_value1   = torch.nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.query_key1     = torch.nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.query_value2   = torch.nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.query_key2     = torch.nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.query_value3   = torch.nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.query_key3     = torch.nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.query_value4   = torch.nn.Conv2d(indim, valdim, kernel_size=1)
        self.query_key4     = torch.nn.Conv2d(indim, keydim, kernel_size=1)

        self.sup_GAP = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.bnn0 = nn.BatchNorm2d(256)
        self.bnn1 = nn.BatchNorm2d(256)
        self.bnn2 = nn.BatchNorm2d(256)
        self.bnn3 = nn.BatchNorm2d(256)
        self.bnn4 = nn.BatchNorm2d(256)

        self.combine = nn.Conv2d(256*2, indim, kernel_size=1, padding=0, stride=1)


    def forward(self, features, supFeatures):
        features = list(features)

        output = []

        h, w = supFeatures.shape[2:]
        ncls = supFeatures.shape[0]  # 15

        sup_value   = self.sup_value(supFeatures)   # [15, 32,  16, 16]
        sup_key     = self.sup_key(supFeatures)     # [15, 128, 16, 16]

        sup_GAP     = self.sup_GAP(supFeatures)     # [15, 256, 1, 1]
        sup_GAP_key = self.sup_GAP_key(sup_GAP)     # [15, 32, 1, 1]

        for idx in range(len(features)):
            feature = features[idx]   # [1, c, H, W]

            bs = feature.shape[0]
            H, W = feature.shape[2:]

            feature = F.interpolate(feature, size=(h, w), mode='bilinear', align_corners=True)  # 下采样为和支持特征同样的尺度

            key_q = eval('self.query_key' + str(idx))(feature).view(bs, 32, -1)  # [bs, 32, H*W]
            val_q = eval('self.query_value' + str(idx))(feature)

            for i in range(bs):
                kq = key_q[i].unsqueeze(0).permute(0, 2, 1)  # [1, h*w, 32]
                vq = val_q[i].unsqueeze(0)                   # [1, 128, h, w]

                sim = torch.matmul(kq, sup_key.view(ncls, 32, -1))
                sim = F.softmax(sim, dim=1)  # [ncls, h*w, h*w]
                val_t_out = torch.matmul(sup_value.view(ncls, 128, -1), sim).view(ncls, 128, h, w)  # [ncls, 128, h, w]

                sim_gap = torch.matmul(kq, sup_GAP_key.view(ncls, 32, -1))  # [ncls, h*w, 1]

                for j in range(ncls):
                    each_atten = sim_gap[j].view(h, -1)                # [h, w]
                    each_atten = torch.sigmoid(each_atten)             # [h, w]
                    each_atten = each_atten.unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]
                    vq_new = vq * each_atten                           # [1, 128, h, w]

                    if (j == 0):
                        final_2 = torch.cat((vq_new, val_t_out[j].unsqueeze(0)), dim=1)  # [1, 256, h, w]
                    else:
                        final_2 += torch.cat((vq_new, val_t_out[j].unsqueeze(0)), dim=1)

                if (i == 0):
                    final_1 = final_2
                else:
                    final_1 = torch.cat((final_1, final_2), dim=0)

            final_1 = F.interpolate(final_1, size=(H, W), mode='bilinear', align_corners=True)      # 恢复为原来的尺寸
            final_1 = eval('self.bnn' + str(idx))(final_1)

            output.append(final_1)        # [1, 256, H, W]

        if self.sum:
            for i in range(len(output)):
                output[i] = self.combine(torch.cat((features[i], output[i]), dim=1))
        output = tuple(output)

        return output