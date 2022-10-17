import os
import os.path as osp

import numpy as np

from .data_set import DataSet


#用于构建train_source, test_source，存储图片目录和数据
def load_data(dataset_path, resolution, dataset, pid_num, pid_shuffle, cache=True):
    seq_dir = list()  # 存放总路径，如：'C:/Users/Administrator/Desktop/out\\124\\nm-06\\180'
    view = list()     # 存放角度，如：018
    label = list()    # 存放ID，如：053

    for _view in sorted(list(os.listdir(dataset_path))):  # 三个循环将目录导入序列 排除包含图片小于五张的目录 完美：11x10x123=13530 实际:13492
        view_path = osp.join(dataset_path, _view)
        for _label in sorted(list(os.listdir(view_path))):
            _seq_dir = osp.join(view_path, _label)
            seqs = os.listdir(_seq_dir)
            if len(seqs) > 0:
                seq_dir.append([_seq_dir])
                label.append(_label)
                view.append(_view)

    pid_fname = osp.join('partition', '{}_{}_{}.npy'.format(
        dataset, pid_num, pid_shuffle))  # 构造信息存储路径
    if not osp.exists(pid_fname):        # 不存在则新建，划定是否打乱顺序，训练集测试集划分
        pid_list = sorted(list(set(label)))
        if pid_shuffle:
            np.random.shuffle(pid_list)
        pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
        os.makedirs('partition', exist_ok=True)
        np.save(pid_fname, pid_list)

    pid_list = np.load(pid_fname, allow_pickle=True)  # 存在则导入
    train_list = pid_list[0]  # 训练集目录信息  ['001',...,'025'] 缺第五组
    test_list = pid_list[1]   # 测试集目录信息  ['026',...,'124']
    train_source = DataSet(   # 构建数据集类，包含目录维度信息并排序，方便调用
        [seq_dir[i] for i, l in enumerate(label) if l in train_list],
        [label[i] for i, l in enumerate(label) if l in train_list],
        [view[i] for i, l in enumerate(label)
         if l in train_list],
        cache, resolution)
    test_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in test_list],
        [label[i] for i, l in enumerate(label) if l in test_list],
        [view[i] for i, l in enumerate(label)
         if l in test_list],
        cache, resolution)

    return train_source, test_source
