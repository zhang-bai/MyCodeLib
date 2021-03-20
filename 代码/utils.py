"""
工具函数
    1.csc2spars_torch
        csc、csr、coo 稀疏格式转 torch.sparse格式
    2.load_gnn_data
        导入 gnn 的 Cora 、Citeseer 、 Pubmed数据集
    3.re_process_adj
        Gnn预处理邻接矩阵，提高稳定性
    4.save_model_Pytorch
        保存 Pytorch 模型
    5.evaluate_Pytorch
        求准确率 accuracy
    6.save_csv
        文件IO
    7.save_pickle
        保存字典等变量
    8.early_stop
        提前停止
    9.cos_similarity
        求余弦相似度
    10.Sparse_dropout_pytorch
        解决grad 变为Fasle问题
    11.save_txt

    12.gen_log
        生成log日志文件
"""


def csc2spars_torch(data_csc):
    # data_csc 为 scipy.sparse.csc格式
    import numpy as np
    import torch
    data_coo = data_csc.tocoo()
    values = data_coo.data.astype(np.float32)
    indices = np.vstack((data_coo.row, data_coo.col)).astype(np.int32)
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = data_coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def load_gnn_data(data_path, dataset_str):
    import numpy as np
    import pickle as pkl
    import networkx as nx
    import scipy.sparse as sp
    import sys

    def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    def sample_mask(idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

    def sparse_to_tuple(sparse_mx):
        """Convert sparse matrix to tuple representation."""

        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

    def preprocess_features(features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return sparse_to_tuple(features)

    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(data_path+"ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(data_path+"ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train, idx_test, idx_val


def pre_process_adj(adj):
    import numpy as np
    import scipy.sparse as sp

    def sparse_to_tuple(sparse_mx):
        """Convert sparse matrix to tuple representation."""

        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx

    def normalize_adj(adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def preprocess_adj(adj):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
        adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
        return sparse_to_tuple(adj_normalized)


def save_model_Pytorch(model, save_path, Model_structure):
    import torch

    # 方法一， 仅保存模型参数
    # 保存
    torch.save(model.state_dict(), save_path+'\parameter.pkl')
    # 加载
    model = Model_structure()
    model.load_state_dict(torch.load(save_path+'\parameter.pkl'))

    # 方法二， 保存模型结构及参数
    # 保存
    torch.save(model, '\model.pkl')
    # 加载
    model = torch.load('\model.pkl')

    # 方法三， 加载checkpoint
    pass


def evaluate_Pytorch():
    pass


def save_csv():
    import numpy as np


def save_pickle(data,file_name):
    import pickle as pkl

    with open(file_name+'_indices.pkl', "wb") as f:
        pkl.dump(data, f)
    # load
    with open(file_name + '_indices.pkl', "rb") as f:
        data2 = pkl.load(f)


def early_stop():
    #     if train_loss < best_loss:
    #         if val_acc > best_val_acc:
    #             best_val_acc = val_acc
    #             best_acc = test_acc
    #             best_epoch = epoch
    #             best_loss = train_loss
    #             count = 0
    #     count += 1
    #     if count > 500:
    #         # print("best_epoch:{:d} | best_acc:{:.3f} | best_loss:{:.5f}".format(best_epoch,best_acc,best_loss))
    #         break
    # print("best_epoch:{:d} | best_loss:{:.5f} | best_acc:{:.3f}".format(best_epoch, best_loss, best_acc))
    pass


def cos_similarity(vector_a, vector_b):
    import numpy as np
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = np.matmul(vector_a, vector_b.T)   # [140,1433] * [1433,1000] = [140,1000]
    # np.linalg.norm(data,axis=1) 计算每一行的L2范数
    denom = np.matmul(np.linalg.norm(vector_a, axis=1)[:, np.newaxis], np.linalg.norm(vector_b, axis=1)[np.newaxis, :])
    cos = num / denom
    sim = 0.5 + 0.5 * cos  # 将 cos 值 [-1,1] 归一化
    # sim = cos
    return sim


def Sparse_dropout_pytorch():
    import torch.nn as nn
    """
    class SparseDropout(nn.Module):

    def __init__(self, p=0.5):
        super(SparseDropout, self).__init__()
        # p is ratio of dropout
        # convert to keep probability
        self.kprob = 1 - p

    def forward(self, x):
        if not self.training:
            return x

        mask = ((torch.rand(x._values().size()) + self.kprob).floor()).type(torch.bool)
        rc = x._indices()[:, mask]
        val = x._values()[mask] * (1.0 / self.kprob)
        return torch.sparse.FloatTensor(rc, val, x.shape).to(x.device)

    # 该版本解决了eval下dropout 问题，主要思路为将输入部分以概率p置零使得对应部分
    # 权重无效，再将val值缩放，使得剩余部分总和与Dropout前一样
    # 但是对于x为 require_grad情况下， grad变为False
    """
    def sparse_dropout(x, rate, noise_shape):
        """

    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).type(torch.bool)

    # 这样改能保留梯度，但是计算速度稍慢
    x = x.to_dense()
    i = x._indices() # [2, 49216]
    v = x[i[0], i[1]]

    # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    out = out * (1./ (1-rate))

    return out


def save_txt():
    f = open("data/model_Weight.txt",'a')  #若文件不存在，系统自动创建。'a'表示可连续写入到文件，保留原内容，在原
    #内容之后写入。可修改该模式（'w+','w','wb'等）
    f.write("hello,sha")  #将字符串写入文件中
    f.write("\n")         #换行  
    # 下图很好的总结了这几种模式：
    # https://www.runoob.com/wp-content/uploads/2013/11/2112205-861c05b2bdbc9c28.png


def gen_log():
import logging  
    logging.debug('this is debug message')  
    logging.info('this is info message')  
    logging.warning('this is warning message')  

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(message)s') 
    #打印结果：WARNING:root:this is warning message 