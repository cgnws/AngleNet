import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bn, **kwargs):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(bn, affine=True)

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x)


class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        x = F.leaky_relu(x, inplace=True)
        return x


class Pool(nn.Module):
    def __init__(self, pad):
        super(Pool, self).__init__()
        self.pool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=pad)

    def forward(self, x):
        x = self.pool2d(x)
        return x


class HPM(nn.Module):
    def __init__(self, in_dim, out_dim, bin_level_num=5):
        super(HPM, self).__init__()
        self.bin_num = [1, 2, 4, 8, 16]
        self.fc_bin1 = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform(
                    torch.zeros(sum(self.bin_num), in_dim, out_dim)))])

        self.fc_bin2 = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform(
                    torch.zeros(sum(self.bin_num), out_dim, 11)))])

    def forward(self, x):
        feature = list()
        n, c, h, w = x.size()
        for num_bin in self.bin_num:
            z = x.view(n, c, num_bin, -1)
            z = z.mean(3)+z.max(3)[0]
            feature.append(z)
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()

        feature = feature.matmul(self.fc_bin1[0])
        feature = feature.matmul(self.fc_bin2[0])
        return feature.permute(1, 0, 2).contiguous()
