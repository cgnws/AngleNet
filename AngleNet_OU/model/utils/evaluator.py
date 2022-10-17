import torch
import torch.nn.functional as F
import numpy as np


def cuda_dist(x, y):  # 求两者距离
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist


def evaluation(data, config, source):
    feature, view, label = data  # [n*110,15872=62*256] [n*110] [n*110] [n*110]
    view_list = list(set(view))
    view_list.sort()

    test_view_set = list(source.view_set)
    test_view_set.sort()
    target_view = [test_view_set.index(l) for l in view]
    featurem = np.argmax(feature, 1)
    viewm = np.array(target_view)
    featurebool = featurem == viewm

    A000=[];A015=[];A030=[];A045=[];A060=[];A075=[];A090=[];A180=[];A195=[];A210=[];A225=[]
    A240=[];A255=[];A270=[]

    for i in range(len(view)):
        if view[i] == "Silhouette_000-00" or view[i] == "Silhouette_000-01":
            A000.append(featurebool[i])
        elif view[i] == "Silhouette_015-00" or view[i] == "Silhouette_015-01":
            A015.append(featurebool[i])
        elif view[i] == "Silhouette_030-00" or view[i] == "Silhouette_030-01":
            A030.append(featurebool[i])
        elif view[i] == "Silhouette_045-00" or view[i] == "Silhouette_045-01":
            A045.append(featurebool[i])
        elif view[i] == "Silhouette_060-00" or view[i] == "Silhouette_060-01":
            A060.append(featurebool[i])
        elif view[i] == "Silhouette_075-00" or view[i] == "Silhouette_075-01":
            A075.append(featurebool[i])
        elif view[i] == "Silhouette_090-00" or view[i] == "Silhouette_090-01":
            A090.append(featurebool[i])
        elif view[i] == "Silhouette_180-00" or view[i] == "Silhouette_180-01":
            A180.append(featurebool[i])
        elif view[i] == "Silhouette_195-00" or view[i] == "Silhouette_195-01":
            A195.append(featurebool[i])
        elif view[i] == "Silhouette_210-00" or view[i] == "Silhouette_210-01":
            A210.append(featurebool[i])
        elif view[i] == "Silhouette_225-00" or view[i] == "Silhouette_225-01":
            A225.append(featurebool[i])
        elif view[i] == "Silhouette_240-00" or view[i] == "Silhouette_240-01":
            A240.append(featurebool[i])
        elif view[i] == "Silhouette_255-00" or view[i] == "Silhouette_255-01":
            A255.append(featurebool[i])
        elif view[i] == "Silhouette_270-00" or view[i] == "Silhouette_270-01":
            A270.append(featurebool[i])

    Angle = [np.sum(np.array(A000))/len(A000), np.sum(np.array(A015))/len(A015), np.sum(np.array(A030))/len(A030),
          np.sum(np.array(A045))/len(A045), np.sum(np.array(A060))/len(A060), np.sum(np.array(A075))/len(A075),
          np.sum(np.array(A090))/len(A090), np.sum(np.array(A180))/len(A180), np.sum(np.array(A195))/len(A195),
          np.sum(np.array(A210))/len(A210), np.sum(np.array(A225))/len(A225), np.sum(np.array(A240))/len(A240),
          np.sum(np.array(A255))/len(A255), np.sum(np.array(A270))/len(A270)]

    return Angle
