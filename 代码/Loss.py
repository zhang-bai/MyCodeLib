"""
神经网络Loss函数
    1.CrossEntropy Pytorch
    2.
"""

import torch
import torch.nn as nn
class CrossEntropy_Pytorch(nn.Module):
    def __init__(self):

        super(CrossEntropy_Pytorch, self).__init__()

    def loss(self, logits, targets):
        # 计算分类CrossEntropy ，输入不为Long 型数值，即标签为一个对应分布
        # 适用于伪标签分类

        g_log_soft = nn.LogSoftmax()
        loss = -torch.mean(torch.sum(g_log_soft(logits) * targets, dim=1))

        return loss



