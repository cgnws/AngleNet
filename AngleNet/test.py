from datetime import datetime
import numpy as np
import argparse

from model.initialization import initialization
from model.utils import evaluation
from config import conf


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--iter', default='80000', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 80000')
parser.add_argument('--batch_size', default='1', type=int,
                    help='batch_size: batch size for parallel test. Default: 1')
parser.add_argument('--cache', default=False, type=boolean_string,
                    help='cache: if set as TRUE all the test data will be loaded at once'
                         ' before the transforming start. Default: FALSE')
opt = parser.parse_args()


# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    if not each_angle:
        result = np.mean(result)
    return result


m = initialization(conf, test=opt.cache)[0]  # 进入初始化，模型加载，图像序列载入

# load model checkpoint of iteration opt.iter
print('Loading the model of iteration %d...' % opt.iter)
m.load(opt.iter)
print('Transforming...')
time = datetime.now()
test = m.transform('test', opt.batch_size)  # 图片跑网络

feature = np.array(test[0])
view = np.array(test[1])
seq_type = np.array(test[2])
label = np.array(test[3])

print('Evaluating...')
NM,BG,CL = evaluation(test, conf['data'], m.test_source)  # [3,11,11,num_rank]
print('Evaluation complete. Cost:', datetime.now() - time)

# Print rank-1 accuracy of the best model
# e.g.
# ===Rank-1 (Include identical-view cases)===
# NM: 95.405,     BG: 88.284,     CL: 72.041
for i in range(1):
    print('===Rank-%d (Include identical-view cases)===' % (i + 1))
    print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
        np.mean(np.array(NM)),
        np.mean(np.array(BG)),
        np.mean(np.array(CL))))


# Print rank-1 accuracy of the best model (Each Angle)
# e.g.
# ===Rank-1 of each angle (Exclude identical-view cases)===
# NM: [90.80 97.90 99.40 96.90 93.60 91.70 95.00 97.80 98.90 96.80 85.80]
# BG: [83.80 91.20 91.80 88.79 83.30 81.00 84.10 90.00 92.20 94.45 79.00]
# CL: [61.40 75.40 80.70 77.30 72.10 70.10 71.50 73.50 73.50 68.40 50.00]
for i in range(1):
    print('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
    print('NM:', np.around(NM, decimals=2))
    print('BG:', np.around(BG, decimals=2))
    print('CL:', np.around(CL, decimals=2))

