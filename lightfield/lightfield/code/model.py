import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.io as scio
from math import sqrt
from numpy import clip
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, angRes, factor):
        super(Net, self).__init__()
        n_blocks, channel = 4, 70
        self.factor = factor
        self.angRes = angRes
        self.FeaExtract1 = FeaExtract(channel,angRes)
        self.FeaExtract2 = FeaExtract(channel, angRes)
        self.FeaExtract3 = FeaExtract(channel, angRes)
        self.FeaExtract4 = FeaExtract(channel, angRes)
        self.conv = nn.Conv2d(channel * 4, angRes * angRes * channel, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)


        self.Fusion1 = FeaExtract2(channel * 4, angRes)
        self.Fusion2 = FeaExtract2(channel * 4, angRes)
        self.Fusion3 = FeaExtract2(channel * 4, angRes)
        self.Fusion4 = FeaExtract2(channel * 4, angRes)
        self.Fusion5 = FeaExtract2(channel * 4, angRes)
        self.Fusion6 = FeaExtract2(channel * 4, angRes)
        self.Fusion7 = FeaExtract2(channel * 4, angRes)
        self.Fusion8 = FeaExtract2(channel * 4, angRes)
        self.FBM = FBM(channel)
        self.UpSample = Upsample(channel, factor)

    def forward(self, x):
        x_e = SAI2MacPI(x,self.angRes)

        x_multi = LFsplit1(x, self.angRes)
        b, n, c, h, w = x_multi.shape
        x_multi = x_multi.contiguous().view(b * n, -1, h, w)
        x_upscale = F.interpolate(x_multi, scale_factor=self.factor, mode='bicubic', align_corners=False)
        _, c, h, w = x_upscale.shape
        x_upscale = x_upscale.unsqueeze(1).contiguous().view(b, -1, c, h, w)


        data_0, data_90, data_45, data_135 = MacPI2EPI(x_e, self.angRes)
        data_0 = self.FeaExtract1(data_0)
        data_90 = self.FeaExtract2(data_90)
        data_45 = self.FeaExtract3(data_45)
        data_135 = self.FeaExtract4(data_135)

        intra_fea = torch.cat((data_0, data_90, data_45, data_135), 1)
        intra_fea = self.Fusion1(intra_fea)
        intra_fea = self.Fusion2(intra_fea)
        intra_fea = self.Fusion3(intra_fea)
        intra_fea = self.Fusion4(intra_fea)
        intra_fea = self.Fusion5(intra_fea)
        intra_fea = self.Fusion6(intra_fea)
        intra_fea = self.Fusion7(intra_fea)
        intra_fea = self.Fusion8(intra_fea)
        intra_fea = self.conv(intra_fea)
        _, _, H, W = intra_fea.shape
        intra_fea = intra_fea.contiguous().view(b, n, -1, H, W)
        intra_fea = self.FBM(intra_fea)

        out_sv = self.UpSample(intra_fea)

        out = FormOutput(out_sv) + FormOutput(x_upscale)

        return out


class Upsample(nn.Module):
    def __init__(self, channel, factor):
        super(Upsample, self).__init__()
        self.upsp = nn.Sequential(
            nn.Conv2d( channel, channel * factor * factor, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(factor),
            nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.contiguous().view(b * n, -1, h, w)
        out = self.upsp(x)
        _, _, H, W = out.shape
        out = out.contiguous().view(b, n, -1, H, W)
        return out

class FBM(nn.Module):
    '''
    Feature Blending
    '''

    def __init__(self, channel):
        super(FBM, self).__init__()
        self.FERB_1 = RB(channel)
        self.FERB_2 = RB(channel)
        self.FERB_3 = RB(channel)
        self.FERB_4 = RB(channel)

        self.att1 = SELayer(channel)
        self.att2 = SELayer(channel)
        self.att3 = SELayer(channel)
        self.att4 = SELayer(channel)


    def forward(self, x):
        b, n, c, h, w = x.shape
        buffer_init = x.contiguous().view(b * n, -1, h, w)
        buffer_1 = self.att1(self.FERB_1(buffer_init))
        buffer_2 = self.att2(self.FERB_2(buffer_1))
        buffer_3 = self.att3(self.FERB_3(buffer_2))
        buffer_4 = self.att4(self.FERB_4(buffer_3))

        buffer = buffer_4.contiguous().view(b, n, -1, h, w)
        return buffer


class FeaExtract(nn.Module):
    def __init__(self, channel,angRes):
        super(FeaExtract, self).__init__()
        self.conv1 = nn.Conv2d(angRes, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x_mv):

        intra_fea_0 = self.conv1(x_mv)
        intra_fea_1 = self.relu(intra_fea_0)
        intra_fea_2 = self.conv2(intra_fea_1)
        intra_fea_3 = self.bn(intra_fea_2)
        intra_fea_4 = self.relu(intra_fea_3)

        return intra_fea_4

class FeaExtract2(nn.Module):
    def __init__(self, channel,angRes):
        super(FeaExtract2, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel//2, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x_mv):

        intra_fea_0 = self.conv1(x_mv)
        intra_fea_1 = self.relu(intra_fea_0)
        intra_fea_2 = self.conv2(intra_fea_1)
        intra_fea_3 = self.bn(intra_fea_2)
        intra_fea_4 = self.relu(intra_fea_3)

        return intra_fea_4

class RB(nn.Module):
    '''
    Residual Block
    '''
    def __init__(self, channel):
        super(RB, self).__init__()
        self.conv01 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.conv02 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        buffer = self.conv01(x)
        buffer = self.lrelu(buffer)
        buffer = self.conv02(buffer)
        return buffer + x
class SELayer(nn.Module):
    '''
    Channel Attention
    '''

    def __init__(self, out_ch, g=16):
        super(SELayer, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.att_c = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // g, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // g, out_ch, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, fm):
        ##channel
        fm_pool = F.adaptive_avg_pool2d(fm, (1, 1))
        fm_max = self.max_pool(fm)
        add = fm_pool + fm_max
        att1 = self.att_c(add)

        fm = fm * att1
        return fm

def ChannelSplit(input):
    _, C, _, _ = input.shape
    c = C // 4
    output_1 = input[:, :c, :, :]
    output_2 = input[:, c:, :, :]
    return output_1, output_2

def LFsplit1(data, angRes):
    b, _, H, W = data.shape
    h = int(H / angRes)
    w = int(W / angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u * h:(u + 1) * h, v * w:(v + 1) * w])

    data_st = torch.stack(data_sv, dim=1)
    return data_st


def MacPI2SAI(x, angRes):
    out = []
    for i in range(angRes):
        out_h = []
        for j in range(angRes):
            out_h.append(x[:, :, i::angRes, j::angRes])
        out.append(torch.cat(out_h, 3))
    out = torch.cat(out, 2)
    return out


def MacPI2EPI(x, angRes):
    data_0 = []
    data_90 = []
    data_45 = []
    data_135 = []

    index_center = int(angRes // 2)
    for i in range(0, angRes, 1):
        img_tmp = x[:, :, index_center::angRes, i::angRes]
        data_0.append(img_tmp)
    data_0 = torch.cat(data_0, 1)

    for i in range(0, angRes, 1):
        img_tmp = x[:, :, i::angRes, index_center::angRes]
        data_90.append(img_tmp)
    data_90 = torch.cat(data_90, 1)

    for i in range(0, angRes, 1):
        img_tmp = x[:, :, i::angRes, i::angRes]
        data_45.append(img_tmp)
    data_45 = torch.cat(data_45, 1)

    for i in range(0, angRes, 1):
        img_tmp = x[:, :, i::angRes, angRes - i - 1::angRes]
        data_135.append(img_tmp)
    data_135 = torch.cat(data_135, 1)

    return data_0, data_90, data_45, data_135


def SAI2MacPI(x, angRes):
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=3))
    out = torch.cat(tempU, dim=2)
    return out

def FormOutput(intra_fea):
    b, n, c, h, w = intra_fea.shape
    angRes = int(sqrt(n + 1))
    out = []
    kk = 0
    for u in range(angRes):
        buffer = []
        for v in range(angRes):
            buffer.append(intra_fea[:, kk, :, :, :])
            kk = kk + 1
        buffer = torch.cat(buffer, 3)
        out.append(buffer)
    out = torch.cat(out, 2)

    return out


if __name__ == "__main__":
    net = Net(5, 4)
    from thop import profile

    input = torch.randn(1, 1, 160, 160)
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))