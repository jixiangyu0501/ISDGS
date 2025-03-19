import torch
import torch.nn as nn
import torch.nn.functional as F
from BBB.network.discriminator import *


# 3D-空谱特征随机化
class Spa_Spe_Randomization(nn.Module):
    def __init__(self, eps=1e-5, device=0):
        super().__init__()
        self.eps = eps
        # self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(device)  # 定义一个可学习的参数，并初始化
        self.alpha = 0.5
        self.cnet = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 32, 1, 1, 0))
        self.snet = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 32, 1, 1, 0))
        self.uncompress = nn.Conv2d(32, 256, 1, 1, 0)

    def forward(self, x, ):
        N, C, L, H, W = x.size()
        if self.training:
            x = x.view(N, C, L, -1)


                # x = x.view(N, C, -1)
                # mean = x.mean(-1, keepdim=True)
                # var = x.var(-1, keepdim=True)

                # x = (x - mean) / (var + self.eps).sqrt()

            idx_swap = torch.randperm(N)
            cF_nor = nor_mean_std(x)
            cF_nr, cmean = nor_mean(x)
            sF_nor, smean = nor_mean(x[idx_swap])
            # cF = self.cnet(cF_nor)
            # sF = self.snet(sF_nor)
            cF = cF_nor
            sF = sF_nor
            b, c, w, h = cF.size()
            c_cov = calc_cov(cF)
            s_cov = calc_cov(sF)
            s_cov = self.alpha * c_cov + (1 - self.alpha) * s_cov
            smean = self.alpha * cmean + (1 - self.alpha) * smean
            gF = torch.bmm(s_cov, cF.flatten(2, 3)).view(b, c, w, h)
            # gF = self.uncompress(gF)
            gF = gF + smean.expand(cF_nor.size())
                # return gF
                # mean = self.alpha * mean + (1 - self.alpha) * mean[idx_swap]  # 从batch中选择随机化均值和方差
                # var = self.alpha * var + (1 - self.alpha) * var[idx_swap]

                # x = x * (var + self.eps).sqrt() + mean

            x = gF.view(N, C, L, H, W)

        return x, idx_swap


class Generator_3DCNN_SupCompress_pca(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], device=0, dim1=8, dim2=16):
        super().__init__()

        self.patch_size = imsize[0]

        self.n_channel = dim2
        self.n_pca = dim1

        # 2D_CONV
        self.conv_pca = nn.Conv2d(imdim, self.n_pca, 1, 1)

        self.inchannel = self.n_pca

        # 3D_CONV
        self.conv1 = nn.Conv3d(in_channels=1,
                               out_channels=self.n_channel,
                               kernel_size=(3, 3, 3))

        # 3D空谱随机化
        self.Spa_Spe_Random = Spa_Spe_Randomization(device=device)

        #
        self.conv6 = nn.ConvTranspose3d(in_channels=self.n_channel, out_channels=1, kernel_size=(3, 3, 3))

        # 2D_CONV
        self.conv_inverse_pca = nn.Conv2d(self.n_pca, imdim, 1, 1)

    def forward(self, x):
        x = self.conv_pca(x)

        x = x.reshape(-1, self.patch_size, self.patch_size, self.inchannel, 1)  # (256,48,13,13,1)转换输入size,适配Conv3d输入
        x = x.permute(0, 4, 3, 1, 2)  # (256,1,48,13,13)

        x = F.relu(self.conv1(x))

        x, idx_swap = self.Spa_Spe_Random(x)

        x = torch.sigmoid(self.conv6(x))

        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(-1, self.inchannel, self.patch_size, self.patch_size)

        x = self.conv_inverse_pca(x)
        return x


# 3D-空谱特征随机化
# class Spa_Spe_Randomization(nn.Module):
#     def __init__(self, eps=1e-5, device=0):
#         super().__init__()
#         self.eps = eps
#         self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(device)
#         self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(device)# 定义一个可学习的参数，并初始化
#         self.cnet = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0),
#                                   nn.ReLU(inplace=True),
#                                   nn.Conv2d(128, 32, 1, 1, 0))
#         self.snet = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0),
#                                   nn.ReLU(inplace=True),
#                                   nn.Conv2d(128, 32, 1, 1, 0))
#         # self.cnet = nn.Sequential(nn.Conv2d(256, 32, 1, 1, 0),
#         #                           nn.ReLU(inplace=True))
#         # self.snet = nn.Sequential(nn.Conv2d(256, 32, 1, 1, 0),
#         #                           nn.ReLU(inplace=True))
#         self.uncompress = nn.Conv2d(32, 256, 1, 1, 0)
#
#     def forward(self, x, ):
#         N, C, H, W = x.size()
#         if self.training:
#             # x = x.view(N, C, -1)
#             # mean = x.mean(-1, keepdim=True)
#             # var = x.var(-1, keepdim=True)
#
#             # x = (x - mean) / (var + self.eps).sqrt()
#
#             idx_swap = torch.randperm(N)
#             cF_nor = nor_mean_std(x)
#             cF_nr, cmean = nor_mean(x)
#             sF_nor, smean = nor_mean(x[idx_swap])
#             cF = self.cnet(cF_nor)
#             sF = self.snet(sF_nor)
#             b, c, w, h = cF.size()
#             c_cov = calc_cov(cF)
#             s_cov = calc_cov(sF)
#             s_cov = self.alpha * c_cov + (1 - self.alpha) * s_cov
#             smean = self.beta * cmean + (1 - self.beta) * smean
#             gF = torch.bmm(s_cov, cF.flatten(2, 3)).view(b, c, w, h)
#             gF = self.uncompress(gF)
#             gF = gF + smean.expand(cF_nor.size())
#             # return gF
#             # mean = self.alpha * mean + (1 - self.alpha) * mean[idx_swap]  # 从batch中选择随机化均值和方差
#             # var = self.alpha * var + (1 - self.alpha) * var[idx_swap]
#
#             # x = x * (var + self.eps).sqrt() + mean
#             x = gF.view(N, C, H, W)
#
#         return x, idx_swap
#
#
# class Generator_3DCNN_SupCompress_pca(nn.Module):
#     def __init__(self, imdim=3, imsize=[13, 13], device=0, dim1=128, dim2=256):
#         super().__init__()
#
#         self.patch_size = imsize[0]
#
#         self.n_channel = dim2
#         self.n_pca = dim1
#         self.dim2 = dim2
#         self.dim1 = dim1
#         self.indim = imdim
#
#         # 2D_CONV
#         self.conv2 = nn.Conv2d(self.dim1, self.dim2, 3, 1)
#         self.conv3 = nn.ConvTranspose2d(in_channels=self.dim2, out_channels=self.dim1, kernel_size=3,
#                                         stride=1)
#
#         self.inchannel = self.n_pca
#
#         # 3D_CONV
#         self.conv1 = nn.Conv2d(in_channels=self.indim,
#                                out_channels=self.dim1,
#                                kernel_size=1)
#
#         # 3D空谱随机化
#         self.Spa_Spe_Random = Spa_Spe_Randomization(device=device)
#
#         #
#         self.conv4 = nn.ConvTranspose2d(in_channels=self.dim1, out_channels=self.indim, kernel_size=1)
#
#         # 2D_CONV
#         self.conv_inverse_pca = nn.Conv2d(self.n_pca, imdim, 1, 1)
#
#     def forward(self, x):
#         # x = x.reshape(-1, self.patch_size, self.patch_size, self.indim, 1)  # (256,51,13,13,1)转换输入size,适配Conv3d输入
#         # x = x.permute(0, 4, 3, 1, 2)  # (256,1,51,13,13)
#         #
#         x = F.relu(self.conv1(x))  # (256,64,13,13)
#         # x = x.reshape(-1, self.dim2 * (self.indim - 2), self.patch_size - 2, self.patch_size - 2)
#         x = F.relu(self.conv2(x))  # (256,64,11,11)
#         x, idx_swap = self.Spa_Spe_Random(x)
#         #
#         x = F.relu(self.conv3(x))  # (256,8*49,11,11)
#         # x = x.reshape(-1, self.dim2, self.indim - 2, self.patch_size - 2, self.patch_size - 2)  # (256,8,49,11,11)
#         x = F.relu(self.conv4(x))                 # (256,1,51,13,13)
#         # x = x.permute(0, 2, 3, 4, 1)                     # (256,51,13,13,1)
#         # x = x.reshape(-1, self.indim, self.patch_size, self.patch_size)
#
#         return x


def nor_mean_std(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    nor_feat = (feat - mean.expand(size)) / std.expand(size)
    return nor_feat


def nor_mean(feat):
    size = feat.size()
    mean = calc_mean(feat)
    nor_feat = feat - mean.expand(size)
    return nor_feat, mean


def calc_cov(feat):
    feat = feat.flatten(2, 3)
    f_cov = torch.bmm(feat, feat.permute(0, 2, 1)).div(feat.size(2))
    return f_cov


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def calc_mean(feat):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean
