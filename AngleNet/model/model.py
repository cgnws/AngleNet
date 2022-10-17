import math
import os
import os.path as osp
import random
import sys
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata
import torch.nn.functional as F

from .network import TripletLoss, SetNet
from .utils import TripletSampler


class Model:
    def __init__(self,
                 hidden_dim, # 全连接层最终特征维度
                 lr,  # 学习率
                 hard_or_full_trip,
                 margin,         # 计算两个向量之间的相似度，当两个向量之间的距离大于margin，则loss为正，小于margin，loss为0。
                 num_workers,    # 进程数
                 batch_size,     # 小批次数目
                 restore_iter,   # 从第几次开始迭代
                 total_iter,     # 迭代数目
                 save_name,   # 'GaitSet_CASIA-B_24_False_256_0.2_128_full_30'
                 train_pid_num,  # 测试集训练集划分
                 frame_num,      # 图片最底层目录编号123x10x11
                 model_name,     # 'GaitSet'
                 train_source,   # 数据集，包含目录和图片数据
                 test_source,
                 img_size=64):

        self.save_name = save_name
        self.train_pid_num = train_pid_num
        self.train_source = train_source
        self.test_source = test_source

        self.hidden_dim = hidden_dim
        self.lr = lr
        self.hard_or_full_trip = hard_or_full_trip
        self.margin = margin
        self.frame_num = frame_num
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model_name = model_name
        self.P, self.M = batch_size

        self.restore_iter = restore_iter
        self.total_iter = total_iter

        self.img_size = img_size

        self.encoder = SetNet(self.hidden_dim).float()  # 网络模型编码，网络初始化
        self.encoder = nn.DataParallel(self.encoder)    # GPU并行
        self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
        self.triplet_loss = nn.DataParallel(self.triplet_loss)
        self.encoder.cuda()  # 指定在哪块显卡上运行
        self.triplet_loss.cuda()

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.encoder.parameters()),
                                    lr=self.lr)

        self.hard_loss_metric = []
        self.full_loss_metric = []
        self.full_loss_num = []
        self.dist_list = []
        self.mean_dist = 0.01
        self.entropyloss = []

        self.sample_type = 'all'

    def collate_fn(self, batch):  # 可以将图片存储在一个list中，计算资源分配到gpu
        batch_size = len(batch)  # batch为batch_size张图片数据，格式为[图片，序号，角度，行走条件，对象编号]
        feature_num = len(batch[0][0])
        seqs = [batch[i][0] for i in range(batch_size)]  # 图片数据 [batch_size:8, 1, frame:110, img_y:64, img_x:44]
        frame_sets = [batch[i][1] for i in range(batch_size)]   # 一个视频包含的图片序号
        view = [batch[i][2] for i in range(batch_size)]
        seq_type = [batch[i][3] for i in range(batch_size)]
        label = [batch[i][4] for i in range(batch_size)]
        batch = [seqs, view, seq_type, label, None]  # 去掉图片序号

        def select_frame(index):  # random则打乱样本
            sample = seqs[index]
            frame_set = frame_sets[index]
            if self.sample_type == 'random':  # 打乱顺序
                frame_id_list = random.choices(frame_set, k=self.frame_num)  # 随机选取frame_num张图片序号
                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:  # 保持原样
                _ = [feature.values for feature in sample]
            return _

        seqs = list(map(select_frame, range(len(seqs))))  # 实质为化繁为简，精简数据格式 [batch_size,1,feature_num,64,44]

        if self.sample_type == 'random':

            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]  # [1,batch_size,feature_num,64,44]
        else:
            gpu_num = min(torch.cuda.device_count(), batch_size)
            batch_per_gpu = math.ceil(batch_size / gpu_num)  # 向上取整，每个gpu分配的图片
            batch_frames = [[     # 该视频图片个数
                                len(frame_sets[i])
                                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                if i < batch_size
                                ] for _ in range(gpu_num)]
            if len(batch_frames[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(batch_frames[-1])):
                    batch_frames[-1].append(0)
            max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
            seqs = [[
                        np.concatenate([
                                           seqs[i][j]
                                           for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                           if i < batch_size
                                           ], 0) for _ in range(gpu_num)]
                    for j in range(feature_num)]
            seqs = [np.asarray([
                                   np.pad(seqs[j][_],
                                          ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                                          'constant',
                                          constant_values=0)
                                   for _ in range(gpu_num)])
                    for j in range(feature_num)]
            batch[4] = np.asarray(batch_frames)

        batch[0] = seqs
        return batch

    def fit(self):  # train.py
        if self.restore_iter != 0:
            self.load(self.restore_iter)

        # c= list(self.encoder.named_parameters())
        # for i, p in enumerate(self.encoder.parameters()):
        #     if i == 0:
        #         p.requires_grad = False

        self.encoder.train()  # 指定当前模型为训练
        self.sample_type = 'random'
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        triplet_sampler = TripletSampler(self.train_source, self.batch_size)
        train_loader = tordata.DataLoader(  # Dataloader的处理逻辑是先通过Dataset类里面的__getitem__函数获取单个的数据，
            dataset=self.train_source,      # 然后组合成batch，再使用collate_fn所指定的函数对这个batch做一些操作
            batch_sampler=triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        train_label_set = list(self.train_source.label_set)  # 人物目录[001,002,...]
        train_label_set.sort()
        train_view_set = list(self.train_source.view_set)
        train_view_set.sort()

        loss_func = nn.CrossEntropyLoss().cuda()

        with open('D:/WORK/reGait/loss.csv', 'w', encoding='utf-8', newline='') as f:
            data = ['hard_loss_metric', 'full_loss_metric',
                    'full_loss_num', 'dist_list', 'restore_iter']
            writer = csv.writer(f)
            writer.writerow(data)

        _time1 = datetime.now()  # 记录时间，开始训练
        for seq, view, seq_type, label, batch_frame in train_loader:
            self.restore_iter += 1
            self.optimizer.zero_grad()

            for i in range(len(seq)):  # 转换为tensor变量，可以使用计算图
                seq[i] = self.np2var(seq[i]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            feature, label_prob = self.encoder(*seq, batch_frame)  # feature:[4,128,18]

            target_label = [train_label_set.index(l) for l in label]  # [1,1,1,1,0,0,0,0] 求label标号 label['002','002','002'，'002'，'001','001','001','001']
            target_label = self.np2var(np.array(target_label)).long()  # 转为tensor

            target_view = [train_view_set.index(l) for l in view]  # [1,1,1,1,0,0,0,0] 求label标号 label['002','002','002'，'002'，'001','001','001','001']
            target_view = self.np2var(np.array(target_view)).long()

            # triplet_feature = feature.permute(1, 0, 2).contiguous()  # [18,4,128]
            # triplet_label = target_label.unsqueeze(0).repeat(triplet_feature.size(0), 1)  # [62,8], 62个[1,1,1,1,0,0,0,0]叠起来
            # triplet_view = target_view.unsqueeze(0).repeat(triplet_feature.size(0), 1)
            # (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num
            #  ) = self.triplet_loss(triplet_feature, triplet_label, triplet_view)  # 维度均为62

            # v = F.one_hot(target_view,11)
            # triplet_feature = torch.mean(triplet_feature, dim=0)
            loss = loss_func(feature, target_view)

            # if self.hard_or_full_trip == 'hard':
            #     loss = hard_loss_metric.mean()
            # elif self.hard_or_full_trip == 'full':
            #     loss = full_loss_metric.mean()

            # self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())   # 画损失曲线图用
            # self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())
            # self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())
            # self.dist_list.append(mean_dist.mean().data.cpu().numpy())
            self.entropyloss.append(loss.mean().data.cpu().numpy())

            if loss > 1e-9:
                loss.backward()
                self.optimizer.step()

            if self.restore_iter % 1000 == 0:
                print(datetime.now() - _time1)
                _time1 = datetime.now()

            if self.restore_iter % 100 == 0:
                with open('D:/WORK/reGait/loss.csv', 'a', encoding='utf-8') as f:
                    # data = [str(np.mean(self.hard_loss_metric)), str(np.mean(self.full_loss_metric)),
                    #         str(np.mean(self.full_loss_num)), str(np.mean(self.dist_list)), str(self.restore_iter)]
                    data = [str(np.mean(self.entropyloss)), str(self.restore_iter)]
                    writer = csv.writer(f)
                    writer.writerow(data)
                print(feature.size())

            if self.restore_iter % 100 == 0:
                self.save()  # 保存网络模型及参数
                print('iter {}:'.format(self.restore_iter), end='')
                # print(', hard_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric)), end='')
                # print(', full_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric)), end='')
                # print(', full_loss_num={0:.8f}'.format(np.mean(self.full_loss_num)), end='')
                # self.mean_dist = np.mean(self.dist_list)
                # print(', mean_dist={0:.8f}'.format(self.mean_dist), end='')
                print(', loss=%f' % self.entropyloss[0], end='')
                print(', lr=%f' % self.optimizer.param_groups[0]['lr'], end='')
                print(', hard or full=%r' % self.hard_or_full_trip)
                sys.stdout.flush()
                # self.hard_loss_metric = []
                # self.full_loss_metric = []
                # self.full_loss_num = []
                # self.dist_list = []
                self.entropyloss = []

            # Visualization using t-SNE
            # if self.restore_iter % 500 == 0:
            #     pca = TSNE(2)
            #     pca_feature = pca.fit_transform(feature.view(feature.size(0), -1).data.cpu().numpy())
            #     for i in range(self.P):
            #         plt.scatter(pca_feature[self.M * i:self.M * (i + 1), 0],
            #                     pca_feature[self.M * i:self.M * (i + 1), 1], label=label[self.M * i])
            #
            #     plt.show()

            if self.restore_iter == self.total_iter:
                break

    def ts2var(self, x):
        return autograd.Variable(x).cuda()   # variable:x放入计算图中计算，cuda:放在GPU上运行

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))  # numpy中的ndarray转化成pytorch中的tensor

    def transform(self, flag, batch_size=1):  # test.py
        self.encoder.eval()  # 指定当前模型为测试
        source = self.test_source if flag == 'test' else self.train_source
        self.sample_type = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),  # 按顺序对数据采样，返回索引值，迭代值，只能用for调用，每次只返回一个索引值
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()

        for i, x in enumerate(data_loader):  # 取出图片
            seq, view, seq_type, label, batch_frame = x  # [1,105,64,44]  ['000'] ['bg-01'] ['006'] [105](该视频图片数目)
            for j in range(len(seq)):  # 转换为tensor变量，可以使用计算图
                seq[j] = self.np2var(seq[j]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()
            # print(batch_frame, np.sum(batch_frame))

            feature, _ = self.encoder(*seq, batch_frame)  # [1,62,256]  None
            # feature = torch.mean(feature, dim=1)
            n, num_bin = feature.size()  # 1 62
            feature_list.append(feature.view(n, -1).data.cpu().numpy())  # 10852:[1,15872]
            view_list += view
            seq_type_list += seq_type
            label_list += label

        return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list

    def save(self):
        os.makedirs(osp.join('checkpoint', self.model_name), exist_ok=True)
        torch.save(self.encoder.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-encoder.ptm'.format(
                                self.save_name, self.restore_iter)))
        torch.save(self.optimizer.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-optimizer.ptm'.format(
                                self.save_name, self.restore_iter)))

    # restore_iter: iteration index of the checkpoint to load
    def load(self, restore_iter):
        self.encoder.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter))))
        self.optimizer.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter))))
