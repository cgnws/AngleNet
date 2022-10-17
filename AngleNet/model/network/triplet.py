import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


class TripletLoss(nn.Module):
    def __init__(self, batch_size, hard_or_full, margin):
        super(TripletLoss, self).__init__()
        self.batch_size = batch_size
        self.margin = margin

    def forward(self, feature, label, view):
        # feature: [n, m, d], label: [n, m] n:图片分割所得维度 m:batch_size d:特征值
        n, m, d = feature.size()                                              # byte()将数据类型int64转换为unit8，范围0-255  18,4,128

        hp_mask = (label.unsqueeze(1) == label.unsqueeze(2)).bool().view(-1)  # 构建人物匹配矩阵,[m,m]，其中1表示两者相同，0表示不同
        hn_mask = (label.unsqueeze(1) != label.unsqueeze(2)).bool().view(-1)  # [n,m,m] -> n*m*m


        dist = self.batch_dist(feature)  # [62,8,8]
        mean_dist = dist.mean(1).mean(1)  # [62]
        dist = dist.view(-1)
        # hard
        hard_hp_dist = torch.max(torch.masked_select(dist, hp_mask).view(n, m, -1), 2)[0]  # 把mask为1的所有值取出来，即为正项，取最大值，即为正项最远距离 [62,8]
        hard_hn_dist = torch.min(torch.masked_select(dist, hn_mask).view(n, m, -1), 2)[0]  # 把mask为0的所有值取出来，即为负项，取最小值，即为负项最近距离 [62,8]
        hard_loss_metric = F.relu(self.margin + hard_hp_dist - hard_hn_dist).view(n, -1)   # loss = p-n+margin [62,8] relu预防过拟合

        hard_loss_metric_mean = torch.mean(hard_loss_metric, 1)  # [62]

        # non-zero full
        full_hp_dist = torch.masked_select(dist, hp_mask).view(n, m, -1, 1)  # [62,8,4,1]
        full_hn_dist = torch.masked_select(dist, hn_mask).view(n, m, 1, -1)  # [62,8,1,4]
        full_loss_metric = F.relu(self.margin + full_hp_dist - full_hn_dist).view(n, -1)  # [62,128]

        full_loss_metric_sum = full_loss_metric.sum(1)  # [62]
        full_loss_num = (full_loss_metric != 0).sum(1).float()  # [62] 统计损失矩阵不为0的数目，数字越小拟合越好

        full_loss_metric_mean = full_loss_metric_sum / full_loss_num  # [62]
        full_loss_metric_mean[full_loss_num == 0] = 0

        return full_loss_metric_mean, hard_loss_metric_mean, mean_dist, full_loss_num

    def batch_dist(self, x):  # 计算x欧式距离 x:[62,8,256]  即sqrt(a^2 + b^2 - 2*a*b)
        x2 = torch.sum(x ** 2, 2)  # 平方和
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))  # 欧氏距离
        dist = torch.sqrt(F.relu(dist))  # 使用relu是为了将负数变为0，正数不变，没啥用
        return dist  # [62,8,8],62个比较维度，8个对象相互之间比较，元素为相互之间的距离值
