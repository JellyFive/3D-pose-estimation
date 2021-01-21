"""
This script will use the 2D box from the label rather than from YOLO,
but will still use the neural nets to get the 3D position and plot onto the
image. Press space for next image and escape to quit
"""
from torch_lib.dataset_posenet import *
from contrast_experiment.model_qu import Model
from library.Math import *
from collections import OrderedDict
import torch.nn.functional as F

import os
import cv2
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg, resnet, mobilenet
import numpy as np

def quat_to_euler(q, is_degree=False):
    w, x, y, z = q[0], q[1], q[2], q[3]

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    # yaw = np.arctan2(t3, t4)
    yaw = 0.0

    if is_degree:
        roll = np.rad2deg(roll)
        pitch = np.rad2deg(pitch)
        # yaw = np.rad2deg(yaw)
        yaw = 0.0

    return np.array([roll, pitch, yaw])

def rot_err_arccos(gt_rot, est_rot):

    ans = np.dot(gt_rot, est_rot)

    rot_err_arccos = np.rad2deg(2 * math.acos(np.abs(ans)))

    return rot_err_arccos


def rot_err_single(gt_rot, est_rot):
    ori_out_euler = quat_to_euler(est_rot)
    ori_true_euler = quat_to_euler(gt_rot)

    rot_err_single = np.abs(ori_true_euler - ori_out_euler)
    return rot_err_single


def main():

    weights_path = '/home/lab/Desktop/wzndeep/posenet-build--eular/contrast_experiment/weight/'
    model_lst = [x for x in sorted(
        os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        print('Using previous model %s' % model_lst[-1])
        base_model = mobilenet.mobilenet_v2(pretrained=True)  # 加载模型并设置为预训练模式
        # model = Model(features=base_model).cuda()
        # base_model = MobileNetV3_Large()
        # state_dict = model_dict()
        # base_model.load_state_dict(state_dict)

        model = Model(features=base_model).cuda()
        checkpoint = torch.load(weights_path + '/%s' % model_lst[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # defaults to /eval
    eval_path = '/media/lab/TOSHIBAEXT/BuildingData/testing'
    dataset = Dataset(eval_path)  # 自定义的数据集

    all_images = dataset.all_objects()

    rot_errors_arccos = []
    rot_errors_single = []
    adds = []

    for key in sorted(all_images.keys()):

        start_time = time.time()

        data = all_images[key]

        truth_img = data['Image']
        img = np.copy(truth_img)
        objects = data['Objects']
        cam_to_img = data['Calib']

        for detectedObject in objects:
            label = detectedObject.label
            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img

            input_tensor = torch.zeros([1, 3, 224, 224]).cuda()
            input_tensor[0, :, :, :] = input_img
            input_tensor.cuda()

            ori_out = model(input_tensor)

            ori_out = F.normalize(ori_out, p=2, dim=1)
            ori_out = ori_out.squeeze(0).detach().cpu().numpy()

            gt_rot = label['Qu']

            # 四元数的反余弦距离
            rot_errors_arccos.append(rot_err_arccos(gt_rot, ori_out))
            # 单独方向的误差
            rot_errors_single.append(rot_err_single(gt_rot, ori_out))


        # print('Estimated patch|Truth patch: {:.3f}/{:.3f}'.format(
        #     patch, r_x))
        # print(
        #     'Estimated yaw|Truth yaw: {:.3f}/{:.3f}'.format(yaw, r_y))

    mean_rot_error_arccos = np.mean(rot_errors_arccos)
    mean_rot_error_single = np.mean(rot_errors_single, axis=0)
    # mean_add = np.mean(adds)

    print('=' * 50)
    print('Got %s poses in %.3f seconds\n' %
          (len(objects), time.time() - start_time))
    print("\tMean Rotation Error Norm: {:.3f}".format(mean_rot_error_arccos))
    print("\tMean Rotation Errors: patch: {:.3f}, yaw: {:.3f}, roll: {:.3f}".format(
        np.rad2deg(mean_rot_error_single[0]), np.rad2deg(mean_rot_error_single[1]),
        np.rad2deg(mean_rot_error_single[2])))
    # print("\tMean ADD: {:.3f}".format(mean_add))


if __name__ == '__main__':
    main()