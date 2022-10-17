conf = {
    "WORK_PATH": "D:/WORK/AngleNet/work",                     # 工作路径，存放训练结果和相关设置
    "CUDA_VISIBLE_DEVICES": "0",                               # 显卡数目，GPU为0,1,2,3
    "data": {
        'dataset_path': "D:/WORK/reGait/input_64",  # 图片输入路径
        'resolution': [64, 64],                                    # 分辨率 64x64
        'dataset': 'CASIA-B',                                  # 数据库 CASIA-B 和 OU-MVLP
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 4,                                         # 训练集测试集划分数目，共124个人，24代表24个训练集100个测试集(缺第五个人，所以24:99)
        'pid_shuffle': False,                                  # 是否打乱排序，最低序列为角度，不能打乱角度内图片顺序
    },
    "model": {
        'hidden_dim': 256,                                     # 隐藏层维度,全连接层的特征维度
        'lr': 1e-4,                                            # 学习率
        'hard_or_full_trip': 'full',                           # 难/全 样本采样三元组损失 hard代表positive取最大值，negative取最小值，增加识别困难程度
        'batch_size': (2, 2),                                 # (p,k)p个身份的人，每个人身上取k张图片
        'restore_iter': 0,                                   # 读取之前的训练数据，比如设为200，则从200次迭代开始训练
        'total_iter': 500,                                   # 训练迭代次数
        'margin': 0.2,                         # 计算两个向量之间的相似度，当两个向量之间的距离大于margin，则loss为正，小于margin，loss为0。
        'num_workers': 0,                                      # 进程数，可多开进程加速
        'frame_num': 30,                                      # 每一次训练取出的图片，比如 002-cl01-000 包含110张图片，我们随机选取其中30张进行训练
        'model_name': 'AngleNet',
    },
}
