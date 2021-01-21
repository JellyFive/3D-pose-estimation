"""
This script will use the 2D box from the label rather than from YOLO,
but will still use the neural nets to get the 3D position and plot onto the
image. Press space for next image and escape to quit
"""
from torch_lib.dataset_posenet import *
from contrast_experiment.model_six import Model
from library.Math import *
from library.evaluate import rot_error, trans_error, add_err
from torch_lib.mobilenetv3_old import MobileNetV3_Large
from collections import OrderedDict

import os
import cv2
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg, resnet, mobilenet
import numpy as np

# 获取mobilenetv_3的预训练模型参数。去掉'module'
def model_dict():
        model = torch.load("/home/lab/Desktop/wzndeep/posenet-build--eular/torch_lib/mbv3_large.old.pth.tar", map_location='cpu')
        weight = model["state_dict"]
        new_state_dict = OrderedDict()
        for k,v in weight.items():
            name = k[7:]
            new_state_dict[name] = v
        return new_state_dict


def main():

    weights_path = '/home/lab/Desktop/wzndeep/posenet-build--eular/contrast_experiment/weights/'
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

    trans_errors_norm = []
    trans_errors_single = []
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

            rotation = model(input_tensor)
            rotation = rotation.cpu().data.numpy()
            print(rotation)

            patch_sin = rotation[0][0]
            patch_cos = rotation[0][1]
            yaw_sin = rotation[0][2]
            yaw_cos = rotation[0][3]
            patch = np.arctan2(patch_sin, patch_cos)
            yaw = np.arctan2(yaw_sin, yaw_cos)
            if yaw >= 1.57:
                yaw -= 1.57

            roll = 0.

            r_x = label['Patch']
            r_y = label['Yaw']
            r_z = label['Roll']

            gt_rot = np.array([r_x, r_y, r_z])
            est_rot = np.array([patch, yaw, roll])

            rot_errors = rot_error(gt_rot, est_rot)
            rot_errors_arccos.append(rot_errors[0])
            rot_errors_single.append(rot_errors[1])

        print('Estimated patch|Truth patch: {:.3f}/{:.3f}'.format(
            patch, r_x))
        print(
            'Estimated yaw|Truth yaw: {:.3f}/{:.3f}'.format(yaw, r_y))

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