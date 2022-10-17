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
    # dataset = config['dataset'].split('-')[0]
    feature, view, seq_type, label = data  # [n*110,15872=62*256] [n*110] [n*110] [n*110]
    # label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    # view_num = len(view_list)
    # sample_num = len(feature)

    test_view_set = list(source.view_set)
    test_view_set.sort()
    target_view = [test_view_set.index(l) for l in view]
    featurem = np.argmax(feature, 1)
    viewm = np.array(target_view)
    featurebool = featurem == viewm
    # a=np.sum(featurebool)
    # mean = a/330

    BG_feature = []
    BG_view = []
    NM_feature = []
    NM_view = []
    CL_feature = []
    CL_view = []

    for i in range(len(seq_type)):
        if seq_type[i] == "bg-01" or seq_type[i] == "bg-02":
            BG_feature.append(featurebool[i])
            BG_view.append(view[i])
        elif seq_type[i] == "cl-01" or seq_type[i] == "cl-02":
            CL_feature.append(featurebool[i])
            CL_view.append(view[i])
        else:
            NM_feature.append(featurebool[i])
            NM_view.append(view[i])

    # a1=np.sum(BG_feature)/66
    # a2=np.sum(CL_feature)/66
    # a3=np.sum(NM_feature)/198

    NM_0=[];NM_18=[];NM_36=[];NM_54=[];NM_72=[];NM_90=[];NM_108=[];NM_126=[];NM_144=[];NM_162=[];NM_180=[]
    BG_0=[];BG_18=[];BG_36=[];BG_54=[];BG_72=[];BG_90=[];BG_108=[];BG_126=[];BG_144=[];BG_162=[];BG_180=[]
    CL_0=[];CL_18=[];CL_36=[];CL_54=[];CL_72=[];CL_90=[];CL_108=[];CL_126=[];CL_144=[];CL_162=[];CL_180=[]


    for i in range(len(NM_view)):
        if NM_view[i] == "000":
            NM_0.append(NM_feature[i])
        elif NM_view[i] == "018":
            NM_18.append(NM_feature[i])
        elif NM_view[i] == "036":
            NM_36.append(NM_feature[i])
        elif NM_view[i] == "054":
            NM_54.append(NM_feature[i])
        elif NM_view[i] == "072":
            NM_72.append(NM_feature[i])
        elif NM_view[i] == "090":
            NM_90.append(NM_feature[i])
        elif NM_view[i] == "108":
            NM_108.append(NM_feature[i])
        elif NM_view[i] == "126":
            NM_126.append(NM_feature[i])
        elif NM_view[i] == "144":
            NM_144.append(NM_feature[i])
        elif NM_view[i] == "162":
            NM_162.append(NM_feature[i])
        elif NM_view[i] == "180":
            NM_180.append(NM_feature[i])

    for i in range(len(BG_view)):
        if BG_view[i] == "000":
            BG_0.append(BG_feature[i])
        elif BG_view[i] == "018":
            BG_18.append(BG_feature[i])
        elif BG_view[i] == "036":
            BG_36.append(BG_feature[i])
        elif BG_view[i] == "054":
            BG_54.append(BG_feature[i])
        elif BG_view[i] == "072":
            BG_72.append(BG_feature[i])
        elif BG_view[i] == "090":
            BG_90.append(BG_feature[i])
        elif BG_view[i] == "108":
            BG_108.append(BG_feature[i])
        elif BG_view[i] == "126":
            BG_126.append(BG_feature[i])
        elif BG_view[i] == "144":
            BG_144.append(BG_feature[i])
        elif BG_view[i] == "162":
            BG_162.append(BG_feature[i])
        elif BG_view[i] == "180":
            BG_180.append(BG_feature[i])

    for i in range(len(CL_view)):
        if CL_view[i] == "000":
            CL_0.append(CL_feature[i])
        elif CL_view[i] == "018":
            CL_18.append(CL_feature[i])
        elif CL_view[i] == "036":
            CL_36.append(CL_feature[i])
        elif CL_view[i] == "054":
            CL_54.append(CL_feature[i])
        elif CL_view[i] == "072":
            CL_72.append(CL_feature[i])
        elif CL_view[i] == "090":
            CL_90.append(CL_feature[i])
        elif CL_view[i] == "108":
            CL_108.append(CL_feature[i])
        elif CL_view[i] == "126":
            CL_126.append(CL_feature[i])
        elif CL_view[i] == "144":
            CL_144.append(CL_feature[i])
        elif CL_view[i] == "162":
            CL_162.append(CL_feature[i])
        elif CL_view[i] == "180":
            CL_180.append(CL_feature[i])



    NM = [np.sum(np.array(NM_0))/len(NM_0), np.sum(np.array(NM_18))/len(NM_18), np.sum(np.array(NM_36))/len(NM_36),
          np.sum(np.array(NM_54))/len(NM_54), np.sum(np.array(NM_72))/len(NM_72), np.sum(np.array(NM_90))/len(NM_90),
          np.sum(np.array(NM_108))/len(NM_108), np.sum(np.array(NM_126))/len(NM_126), np.sum(np.array(NM_144))/len(NM_144),
          np.sum(np.array(NM_162))/len(NM_162), np.sum(np.array(NM_180))/len(NM_180)]
    BG = [np.sum(np.array(BG_0))/len(BG_0), np.sum(np.array(BG_18))/len(BG_18), np.sum(np.array(BG_36))/len(BG_36),
          np.sum(np.array(BG_54))/len(BG_54), np.sum(np.array(BG_72))/len(BG_72), np.sum(np.array(BG_90))/len(BG_90),
          np.sum(np.array(BG_108))/len(BG_108), np.sum(np.array(BG_126))/len(BG_126), np.sum(np.array(BG_144))/len(BG_144),
          np.sum(np.array(BG_162))/len(BG_162), np.sum(np.array(BG_180))/len(BG_180)]
    CL = [np.sum(np.array(CL_0))/len(CL_0), np.sum(np.array(CL_18))/len(CL_18), np.sum(np.array(CL_36))/len(CL_36),
          np.sum(np.array(CL_54))/len(CL_54), np.sum(np.array(CL_72))/len(CL_72), np.sum(np.array(CL_90))/len(CL_90),
          np.sum(np.array(CL_108))/len(CL_108), np.sum(np.array(CL_126))/len(CL_126), np.sum(np.array(CL_144))/len(CL_144),
          np.sum(np.array(CL_162))/len(CL_162), np.sum(np.array(CL_180))/len(CL_180)]




    return NM,BG,CL
