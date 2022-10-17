import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .basic_blocks1 import Conv, Pool, ReLU, HPM


class SetNet(nn.Module):
    def __init__(self, hidden_dim):
        super(SetNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_frame = None

        _in_channels = 1
        _channels = [32, 64, 128]
        self.set_layer0 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        # V纵向卷积核，H横向卷积核，F全卷积核
        self.set_layer1F = Conv(_in_channels, _channels[0], (3, 3), stride=1, padding=(1, 1), bn=_channels[0])
        self.set_layer2F = Conv(_channels[0], _channels[0], (3, 3), stride=1, padding=(1, 1), bn=_channels[0])
        self.set_layer3F = Conv(_channels[0], _channels[1], (3, 3), stride=1, padding=(1, 1), bn=_channels[1])
        self.set_layer4F = Conv(_channels[1], _channels[1], (3, 3), stride=1, padding=(1, 1), bn=_channels[1])
        self.set_layer5F = Conv(_channels[1], _channels[2], (3, 3), stride=1, padding=(1, 1), bn=_channels[2])
        self.set_layer6F = Conv(_channels[2], _channels[2], (3, 3), stride=1, padding=(1, 1), bn=_channels[2])

        self.sa_layer = Conv(2, 1, 3, stride=1, padding=1, bn=1)

        # self.pool = Pool(0)
        # self.pool_pad1 = Pool(1)
        self.relu = ReLU()

        self.maxpool = nn.MaxPool2d(2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        # self.avgpool2 = nn.AvgPool2d(kernel_size=(1, 6), stride=1, padding=0)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 6), stride=1, padding=0)

        # self.x_hpm = HPM(_channels[-1], hidden_dim)
        self.fc1 = nn.Sequential(nn.Dropout(), nn.Linear(8192, 512), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Dropout(), nn.Linear(512, 512), nn.ReLU(inplace=True))
        self.out = nn.Linear(512, 11)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal(m.weight.data, 1.0, 0.02)
                nn.init.constant(m.bias.data, 0.0)

    def frame_max(self, x):
        if self.batch_frame is None:
            return torch.max(x, 1)
        else:
            _tmp = [
                torch.max(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
            ]
            max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return max_list, arg_max_list

    def forward(self, silho, batch_frame=None):
        # n: batch_size, s: frame_num, k: keypoints_num, c: channel
        if batch_frame is not None:
            batch_frame = batch_frame[0].data.cpu().numpy().tolist()
            _ = len(batch_frame)
            for i in range(len(batch_frame)):
                if batch_frame[-(i + 1)] != 0:
                    break
                else:
                    _ -= 1
            batch_frame = batch_frame[:_]
            frame_sum = np.sum(batch_frame)
            if frame_sum < silho.size(1):
                silho = silho[:, :frame_sum, :, :]
            self.batch_frame = [0] + np.cumsum(batch_frame).tolist()
        n = silho.size(0)
        x = silho.unsqueeze(2)
        n, s, c, h, w = x.size()

        x = x.view(-1, c, h, w)
        del silho

        x = self.set_layer0(x)
        x = self.block1(x, self.set_layer1F, self.set_layer2F)
        x = self.block2(x, self.set_layer3F, self.set_layer4F)
        x = self.block3(x, self.set_layer5F, self.set_layer6F)

        _, c, h, w = x.size()
        x = x.view(n, s, c, h, w)
        x = self.frame_max(x)[0]  # [4,128,16,6]
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        # x = F.softmax(x, dim=1)

        # feature = self.x_hpm(x)
        # feature = F.dropout(feature, p=0.2, training=self.training)

        return x, None

    def block1(self, x, layer1, layer2):
        x = layer1(x)  # [120,32,72,6]
        x = self.relu(x)
        x1 = x
        x = layer2(x)
        x = x + x1
        x = self.relu(x)
        x = self.maxpool(x)
        # x = F.dropout(x, p=0.2, training=self.training)

        return x

    def block2(self, x, layer1, layer2):
        x = layer1(x)  # [120,32,72,6]
        x = self.relu(x)
        x1 = x
        x = layer2(x)
        x = x + x1
        x = self.relu(x)
        x = self.maxpool1(x)
        # x = F.dropout(x, p=0.2, training=self.training)

        return x

    def block3(self, x, layer1, layer2):
        x = layer1(x)  # [120,32,72,6]
        x = self.relu(x)
        x1 = x
        x = layer2(x)
        x = x + x1
        x = self.relu(x)
        x = self.maxpool1(x)
        # x = F.dropout(x, p=0.2, training=self.training)

        return x
