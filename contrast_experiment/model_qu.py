import os
import time
import copy
import torch
import torchvision

# import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
import torch.nn.functional as F


class PoseLoss(nn.Module):
    def __init__(self, device, sq=0.0, learn_beta=False):
        super(PoseLoss, self).__init__()
        self.learn_beta = learn_beta

        if not self.learn_beta:
            self.sx = 0
            self.sq = -6.25

        self.sq = nn.Parameter(torch.Tensor(
            [sq]), requires_grad=self.learn_beta)

        self.loss_print = None

    def forward(self, pred_q, target_q):
        pred_q = F.normalize(pred_q, p=2, dim=1)
        loss_q = F.l1_loss(pred_q, target_q)

        loss = torch.exp(-self.sq) * loss_q + self.sq

        self.loss_print = [loss.item(), loss_q.item()]

        return loss, loss_q.item()


class Model(nn.Module):
    def __init__(self, features=None, fixed_weight=False, dropout_rate=0.0):
        super(Model, self).__init__()
        # feat_in = features.fc.in_features

        # self.features = nn.Sequential(*list(features.children())[:-1])
        # self.base_model = base_model

        self.features = features

        if fixed_weight:
            for param in self.features.parameters():
                param.requires_grad = False

        # self.fc_rotation = nn.Linear(1000, 4, bias=False)

        self.fc_rotation = nn.Sequential(
            # nn.Linear(512, 512),  # resnet101
            nn.Linear(1000, 512),  # mobilenetv2
            # nn.Linear(512 * 7 * 7, 256),  # 全连接函数，将512*7*7连接到256个节点上 vgg19
            nn.ReLU(True),  # 激活函数
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 4)  # to get sin and cos # 角度回归
        )

        init_modules = [self.fc_rotation]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight.data)
                if module.bias is not None:
                    nn.init.constant(module.bias.data, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        rotation = self.fc_rotation(x)
        return rotation
