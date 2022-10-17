import torch.utils.data as tordata
import numpy as np
import os.path as osp
import os
import pickle
import cv2
import xarray as xr


# 将目录信息和图像分辨率集结为source，仅用于train/test_source
class DataSet(tordata.Dataset):  # 数据集类，主体为data和frame_set，负责存储检索图片数据
    def __init__(self, seq_dir, label, seq_type, view, cache, resolution):
        self.seq_dir = seq_dir     # 总目录   dim:2640(train)  10852(test)
        self.view = view           # 三级标签 dim:2640(train)  10852(test)
        self.seq_type = seq_type   # 二级标签 dim:2640(train)  10852(test)
        self.label = label         # 一级标签 dim:2640(train)  10852(test)
        self.cache = cache         # train:False  test:False
        self.resolution = resolution
        self.data_size = len(self.label)  # 2640
        self.data = [None] * self.data_size
        self.frame_set = [None] * self.data_size

        self.label_set = set(self.label)  # set()删除重复项 dim：24(train)  99(test)
        self.seq_type_set = set(self.seq_type)  # dim：10
        self.view_set = set(self.view)          # dim: 11
        _ = np.zeros((len(self.label_set),
                      len(self.seq_type_set),
                      len(self.view_set))).astype('int')
        _ -= 1  # dim:(24,10,11)(train)  (99,10,11)(test)
        self.index_dict = xr.DataArray(   # 带标签的数组，方便各维度信息可视化，此处为目录矩阵
            _,
            coords={'label': sorted(list(self.label_set)),
                    'seq_type': sorted(list(self.seq_type_set)),
                    'view': sorted(list(self.view_set))},
            dims=['label', 'seq_type', 'view'])

        for i in range(self.data_size):   # 目录矩阵元素排序
            _label = self.label[i]
            _seq_type = self.seq_type[i]
            _view = self.view[i]
            self.index_dict.loc[_label, _seq_type, _view] = i  # 查找方式

    def load_all_data(self):  # 改变DataSet的data和frame_set，读取并存储图片，将(64,64,3)修正为(64,4,1)
        for i in range(self.data_size):
            self.load_data(i)

    def load_data(self, index):  # index:图片序号 0-2639
        return self.__getitem__(index)  # 五项输出，仅用于test

    def __loader__(self, path): # 取出图片，宽度缩减，数据类型转换为float，灰度值除以255变为0-1
        return self.img2xarray(
            path)[:, :, :].astype(
            'float32')

    def __getitem__(self, index):  # index:图片序号 0-2639  取出指定序号图片
        # pose sequence sampling
        if not self.cache:  # cache=false, 边训练边读取
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]  # [81,64,44]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]   # [81] 视频图片数目编号
            frame_set = list(set.intersection(*frame_set))  # 取交集去重
        elif self.data[index] is None:  # cache=true,训练前构建source,图片数据直接存入DataSet
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))  # 取交集，这里用于去掉重复
            self.data[index] = data        # loaddata 主体，为DataSet存储数据
            self.frame_set[index] = frame_set
        else:  # 训练过程中直接调用DataSet数据，无需重复取图片
            data = self.data[index]
            frame_set = self.frame_set[index]

        return data, frame_set, self.view[
            index], self.seq_type[index], self.label[index],  # 图片，序号，三层目录坐标

    def img2xarray(self, flie_path):
        imgs = sorted(list(os.listdir(flie_path)))  # 取出图片名称，排序
        frame_list = [np.reshape(                   # 取出图片[64,64,3],转为[64,64]
            cv2.imread(osp.join(flie_path, _img_path)),
            [self.resolution[0], self.resolution[1], -1])[:, :, 0]
                      for _img_path in imgs
                      if osp.isfile(osp.join(flie_path, _img_path))]
        num_list = list(range(len(frame_list)))
        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'img_y', 'img_x'],
        )
        return data_dict   # 图片和序号对应字典

    def __len__(self):
        return len(self.label)
