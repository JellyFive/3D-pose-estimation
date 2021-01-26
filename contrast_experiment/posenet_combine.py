import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tensorboardX import SummaryWriter


def OrientationLoss(orient_batch, orientGT_batch, confGT_batch):

    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]

    # extract just the important bin
    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
    orient_batch = orient_batch[torch.arange(batch_size), indexes]

    theta_diff = torch.atan2(orientGT_batch[:, 1], orientGT_batch[:, 0])
    estimated_theta_diff = torch.atan2(orient_batch[:, 1], orient_batch[:, 0])

    return -1 * torch.cos(theta_diff - estimated_theta_diff).mean()



class Model(nn.Module):
    def __init__(self, features=None, bins=8, w=1):  # 神经网络的基本结构
        super(Model, self).__init__()  # 先运行nn.Module的初始化函数
        self.bins = bins
        self.w = w
        self.features = features
        # self.features = nn.Sequential(*list(features.children())[:-1])

        self.s1 = nn.Linear(1000, 512)
        self.s2 = nn.ReLU(True)
        self.s3 = nn.Dropout()
        self.s4 = nn.Linear(512, 256)

        self.orientation = nn.Linear(256, bins*2)

        self.location = nn.Linear(256, 3)

        self.confidence = nn.Sequential(
            nn.Linear(1000, 512),  # mobilenetv2
            # nn.Linear(512, 512),  # resnet101
            # nn.Linear(512 * 7 * 7, 256),  # vgg19
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, bins),
            # nn.Softmax()
            # nn.Sigmoid()
        )

        self.confidence_yaw = nn.Sequential(
            nn.Linear(1000, 512),  # mobilenetv2
            # nn.Linear(512, 512),  # resnet101
            # nn.Linear(512 * 7 * 7, 256),  # vgg19
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, bins),
            # nn.Softmax()
            # nn.Sigmoid()
        )

    def forward(self, x):  # 前向传播函数，向后传播函数自动生成autograd
        x = self.features(x)  # 512 x 7 x 7
        # view函数将张量x形成一维的向量形式，总特征数是512*7*7，用于后面的全连接层 vgg19
        # x = x.view(-1, 512 * 7 * 7)
        x = x.view(x.size(0), -1)  # resnet101

        patch_s1 = self.s1(x)
        patch_s2 = self.s2(patch_s1)
        patch_s3 = self.s3(patch_s2)
        patch_s4 = self.s4(patch_s3)
        patch_s5 = self.s2(patch_s4)
        patch_s6 = self.s3(patch_s5)
        orientation_patch = self.orientation(patch_s6)
        orientation_patch = orientation_patch.view(-1, self.bins, 2)
        orientation_patch = F.normalize(orientation_patch, dim=2)
        confidence = self.confidence(x)

        yaw_s1 = self.s1(x)
        yaw_s2 = self.s2(yaw_s1)
        yaw_s3 = self.s3(yaw_s2)
        yaw_s4 = self.s4(yaw_s3)
        yaw_s5 = self.s2(yaw_s4)
        yaw_s6 = self.s3(yaw_s5)
        orientation_yaw = self.orientation(yaw_s6)
        orientation_yaw = orientation_yaw.view(-1, self.bins, 2)
        orientation_yaw = F.normalize(orientation_yaw, dim=2)
        confidence_yaw = self.confidence_yaw(x)

        trans_s1 = self.s1(x)
        trans_s2 = self.s2(trans_s1)
        trans_s3 = self.s3(trans_s2)
        trans_s3 = trans_s3 + patch_s3 +yaw_s3
        trans_s4 = self.s4(trans_s3)
        trans_s5 = self.s2(trans_s4)
        trans_s6 = self.s3(trans_s5)
        trans_s6 = trans_s6 + patch_s6 +yaw_s6
        location = self.location(trans_s6)

        return orientation_patch, confidence, orientation_yaw, confidence_yaw, location