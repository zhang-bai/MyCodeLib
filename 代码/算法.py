"""Pytorch

1.筛选Graph相邻三元组下标
"""
class Shaixuan_index():
    import torch
    import numpy as np
    import pickle as pkl
    import networkx as nx
    import scipy.sparse as sp
    import sys
    from pathlib import Path

    # 新建data文件夹存放AC下标
    file_path='./data_AC/'
    file_path = Path(file_path)
    if not file_path.exists():
        file_path.mkdir(parents=True, exist_ok=True)
    file_name = str(file_path)+'/'+patten

    # 筛选index
    adj_g = adj.to_dense()
    adj_g *= torch.ones_like(adj_g).cuda() - torch.eye(adj_g.shape[0]).cuda()
    f_index = torch.arange(0, features.shape[0], 2)
    f_index = f_index.type(torch.long)

    # 只考虑有 label 的样本来选择 idx
    idx_b = idx_train.cpu()
    adj_bbb = adj_g[idx_b]
    idx_ccc = torch.gt(adj_bbb, 0)
    adj_onehop = adj_bbb.nonzero()
    nb_onehop = torch.sum(idx_ccc, dim=1)
    del adj_bbb, idx_ccc

    indices = dict()
    sset = set(idx_train.cpu().numpy())
    indices_train =  dict()
    count_ = 0
    ct_ = 0
    while count_ < nb_onehop.shape[0]:
        tempee = []
        tempe_train = []
        for _ in range(nb_onehop[count_].cpu().numpy()):
            if nb_onehop[count_] == 1:
                ct_ += 1
                continue
            else:
                tt = int(adj_onehop[ct_, 1].cpu().numpy())
                tempee.append(tt)
                indices[int(idx_b[count_])] = tempee
                ct_ += 1
                if tt in sset:
                    tempe_train.append(tt)
                    indices_train[int(idx_b[count_])] = tempe_train

        count_ += 1

    del count_, ct_