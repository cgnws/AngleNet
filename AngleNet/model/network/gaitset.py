import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .basic_blocks import SetBlock, BasicConv2d, HPM


class SetNet(nn.Module):
    def __init__(self, hidden_dim):
        super(SetNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_frame = None

        _in_channels = 1
        _channels = [32, 64, 128]
        self.set_layer1 = SetBlock(BasicConv2d(_in_channels, _channels[0], 5, padding=2, bn=32))
        self.set_layer2 = SetBlock(BasicConv2d(_channels[0], _channels[0], 3, padding=1, bn=32), True)
        self.set_layer3 = SetBlock(BasicConv2d(_channels[0], _channels[1], 3, padding=1, bn=64))
        self.set_layer4 = SetBlock(BasicConv2d(_channels[1], _channels[1], 3, padding=1, bn=64), True)
        self.set_layer5 = SetBlock(BasicConv2d(_channels[1], _channels[2], 3, padding=1, bn=128))
        self.set_layer6 = SetBlock(BasicConv2d(_channels[2], _channels[2], 3, padding=1, bn=128))

        self.gl_layer1 = BasicConv2d(_channels[0], _channels[1], 3, padding=1, bn=64)
        self.gl_layer2 = BasicConv2d(_channels[1], _channels[1], 3, padding=1, bn=64)
        self.gl_layer3 = BasicConv2d(_channels[1], _channels[2], 3, padding=1, bn=128)
        self.gl_layer4 = BasicConv2d(_channels[2], _channels[2], 3, padding=1, bn=128)
        self.gl_pooling = nn.MaxPool2d(2)

        self.gl_hpm = HPM(_channels[-1], hidden_dim)
        self.x_hpm = HPM(_channels[-1], hidden_dim)

        self.out = nn.Linear(256, 11)

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
                torch.max(x[:, self.batch_frame[i]:self.batch_frame[i+1], :, :, :], 1)
                for i in range(len(self.batch_frame)-1)
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
                if batch_frame[-(i+1)] != 0:
                    break
                else:
                    _ -= 1
            batch_frame = batch_frame[:_]
            frame_sum = np.sum(batch_frame)
            if frame_sum < silho.size(1):
                silho = silho[:, :frame_sum, :, :]
            self.batch_frame = [0]+np.cumsum(batch_frame).tolist()
        n = silho.size(0)
        x = silho.unsqueeze(2)
        del silho

        x = self.set_layer1(x)  # [4,30,32,72,6]
        x = self.set_layer2(x)  # [4,30,32,36,3]
        x = F.dropout(x, p=0.2, training=self.training)
        gl = self.gl_layer1(self.frame_max(x)[0])  # [4,64,36,3]
        gl = self.gl_layer2(gl)  # [4,64,36,3]
        gl = self.gl_pooling(gl)  # [4,64,18,1]

        x = self.set_layer3(x)  # [4,30,64,36,3]
        x = self.set_layer4(x)  # [4,30,64,18,1]
        x = F.dropout(x, p=0.3, training=self.training)
        gl = self.gl_layer3(gl+self.frame_max(x)[0])  # [4,128,18,1]
        gl = self.gl_layer4(gl)  # [4,128,18,1]

        x = self.set_layer5(x)  # [4,30,128,18,1]
        x = self.set_layer6(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.frame_max(x)[0]  # [4,128,16,16]
        gl = gl+x  # [4,128,16,16]

        gl_f = self.gl_hpm(gl)  # [4,31,256]
        x_f = self.x_hpm(x)  # [4,31,256]
        feature = torch.cat([gl_f, x_f], 1).mean(1).view(x.size(0),-1)  # [4,256]
        feature = self.out(feature)

        return feature, None