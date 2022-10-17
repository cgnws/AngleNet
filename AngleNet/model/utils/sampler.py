import torch.utils.data as tordata
import random

# 三元组样本提取，先随机选择p个目标，乱序，每个目标随机选取k个视频序列,返回序号   batch_size  (p,k)
class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        while (True):
            sample_indices = list()
            pid_list = random.sample(  # 随机抽batch_size[0]个目标,乱序
                list(self.dataset.label_set),
                self.batch_size[0])
            for pid in pid_list:
                _index = self.dataset.index_dict.loc[pid, :, :].values  # 提取目标目录 10x11
                _index = _index[_index > 0].flatten().tolist()  # 转换为一维数组， 不知为何去掉了0
                _index = random.choices(   # 随机抽batch_size[1]个图片序号，允许重复
                    _index,
                    k=self.batch_size[1])
                sample_indices += _index  # batch_size[0] x batch_size[1] 个序号累计
            yield sample_indices

    def __len__(self):
        return self.dataset.data_size
