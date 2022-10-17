# -*- coding: utf-8 -*-
# @Author  : admin
# @Time    : 2018/11/15
import os
from copy import deepcopy

import numpy as np

from .utils import load_data
from .model import Model


def initialize_data(config, train=False, test=False):  # 初始化数据目录，简单筛查，加载train/test_source , cache=False直接跳过
    print("Initializing data source...")
    train_source, test_source = load_data(**config['data'], cache=(train or test))  # 加载或构建训练和测试数据目录信息:不含图
    if train:
        print("Loading training data...")
        train_source.load_all_data()
    if test:
        print("Loading test data...")
        test_source.load_all_data()
    print("Data initialization complete.")
    return train_source, test_source


def initialize_model(config, train_source, test_source):
    print("Initializing model...")
    data_config = config['data']
    model_config = config['model']
    model_param = deepcopy(model_config)  # 复制， 防止误改
    model_param['train_source'] = train_source
    model_param['test_source'] = test_source
    model_param['train_pid_num'] = data_config['pid_num']
    batch_size = int(np.prod(model_config['batch_size']))  # prod计算数组元素乘积 16x8=128 仅用于save_name
    model_param['save_name'] = '_'.join(map(str, [  # 组合训练集权重文件名称
        model_config['model_name'],
        data_config['dataset'],
        data_config['pid_num'],
        data_config['pid_shuffle'],
        model_config['hidden_dim'],
        model_config['margin'],
        batch_size,
        model_config['hard_or_full_trip'],
        model_config['frame_num'],
    ]))

    m = Model(**model_param)   # model_param构建链表，将数据集中起来传入Model
    print("Model initialization complete.")
    return m, model_param['save_name']


def initialization(config, train=False, test=False):
    print("Initialzing...")
    WORK_PATH = config['WORK_PATH']  # 加载工作目录
    os.chdir(WORK_PATH)  # 改变当前工作目录到指定的路径
    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]  # 加载GPU数目 os.environ['环境变量名称']='环境变量值'
    train_source, test_source = initialize_data(config, train, test)  # 初始化数据，将图片数据载入source
    return initialize_model(config, train_source, test_source)