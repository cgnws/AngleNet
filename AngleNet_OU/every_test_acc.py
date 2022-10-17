from datetime import datetime
import numpy as np
import argparse
import csv

from model.initialization import initialization
from model.utils import evaluation
from config import conf
#############################################返回单人样本#################################################


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--iter', default='80000', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 80000')
parser.add_argument('--batch_size', default='1', type=int,
                    help='batch_size: batch size for parallel test. Default: 1')
parser.add_argument('--cache', default=True, type=boolean_string,
                    help='cache: if set as TRUE all the test data will be loaded at once'
                         ' before the transforming start. Default: FALSE')
opt = parser.parse_args()


# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    if not each_angle:
        result = np.mean(result)
    return result


test_iter = int(opt.iter/100)
m = initialization(conf, test=opt.cache)[0]  # 进入初始化，模型加载，图像序列载入

with open('/home/cgn/Desktop/testacc.csv', 'w', encoding='utf-8', newline='') as f:
    data = ['NM', 'BG', 'CL', 'iter']
    writer = csv.writer(f)
    writer.writerow(data)

for i in range(test_iter):
    iter = (i+1)*100
    # load model checkpoint of iteration opt.iter
    print('Loading the model of iteration %d...' % iter)
    m.load(iter)
    time = datetime.now()
    test = m.transform('test', opt.batch_size)  # 图片跑网络
    acc = evaluation(test, conf['data'])  # [3,11,11,num_rank]
    print('Evaluation complete. Cost:', datetime.now() - time)

    nm_acc = de_diag(acc[0, :, :, 0])
    bg_acc = de_diag(acc[1, :, :, 0])
    cl_acc = de_diag(acc[2, :, :, 0])

    with open('/home/cgn/Desktop/testacc.csv', 'a', encoding='utf-8') as f:
        data = [nm_acc, bg_acc, cl_acc, iter]
        writer = csv.writer(f)
        writer.writerow(data)


