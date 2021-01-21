import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tensorboardX import SummaryWriter


class Model(nn.Module):
    def __init__(self, features=None, bins=8, w=1):  # 神经网络的基本结构
        super(Model, self).__init__()  # 先运行nn.Module的初始化函数
        self.bins = bins
        self.w = w
        self.features = features
        # self.features = nn.Sequential(*list(features.children())[:-1])
        self.orientation = nn.Sequential(
            # nn.Linear(512, 512),  # resnet101
            nn.Linear(1000, 512),  # mobilenetv2
            # nn.Linear(512 * 7 * 7, 256),  # 全连接函数，将512*7*7连接到256个节点上 vgg19
            nn.ReLU(True),  # 激活函数
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 3)           
        )

    def forward(self, x):  # 前向传播函数，向后传播函数自动生成autograd
        x = self.features(x)  # 512 x 7 x 7
        # view函数将张量x形成一维的向量形式，总特征数是512*7*7，用于后面的全连接层 vgg19
        # x = x.view(-1, 512 * 7 * 7)
        x = x.view(x.size(0), -1)
        rotation = self.orientation(x)
        return rotation