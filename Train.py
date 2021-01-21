from torch_lib.dataset_posenet import *
from torch_lib.posenet import Model, OrientationLoss
from torch_lib.mobilenetv3_old import MobileNetV3_Large
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg, resnet, densenet, mobilenet
from torch.utils import data

from tensorboardX import SummaryWriter

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

    # hyper parameters
    epochs = 200
    batch_size = 8
    w = 1
    alpha = 1

    print("Loading all detected objects in dataset...")

    train_path = "/media/lab/TOSHIBAEXT/BuildingData/training"
    dataset = Dataset(train_path)  # 自定义的数据集

    params = {"batch_size": batch_size, "shuffle": True, "num_workers": 6}

    generator = data.DataLoader(dataset, **params)  # 读取Dataset中的数据

    base_model = mobilenet.mobilenet_v2(pretrained=True)  # 加载模型并设置为预训练模式
    # base_model = MobileNetV3_Large()
    # state_dict = model_dict()
    # base_model.load_state_dict(state_dict)

    model = Model(features=base_model).cuda()

    # 选择不同的优化方法
    opt_Momentum = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.9)
    opt_RMSprop = torch.optim.RMSprop(model.parameters(), lr = 0.0001, alpha = 0.9)
    opt_Adam = torch.optim.Adam(model.parameters(), lr = 0.0001, betas= (0.9, 0.99))
    opt_SGD = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    conf_loss_func = nn.CrossEntropyLoss().cuda()
    orient_loss_func = OrientationLoss

    # load any previous weights
    model_path = ("/home/lab/Desktop/wzndeep/posenet-build--eular/weights/")
    latest_model = None
    first_epoch = 0
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    else:
        try:
            latest_model = [
                x for x in sorted(os.listdir(model_path)) if x.endswith(".pkl")
            ][-1]
        except:
            pass

    if latest_model is not None:
        # 解序列化一个pickled对象并加载到内存中
        checkpoint = torch.load(model_path + latest_model)
        # 加载一个state_dict对象，加载模型用于训练或验证
        model.load_state_dict(checkpoint["model_state_dict"])
        opt_SGD.load_state_dict(checkpoint["optimizer_state_dict"])  # 同上
        first_epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]

        print("Found previous checkpoint: %s at epoch %s" %
              (latest_model, first_epoch))
        print("Resuming training....")

    # 训练网络

    total_num_batches = int(len(dataset) / batch_size)

    writer = SummaryWriter(
        "/home/lab/Desktop/wzndeep/posenet-build--eular/runs/")

    for epoch in range(first_epoch + 1, epochs + 1):  # 多批次循环
        curr_batch = 0
        passes = 0
        for local_batch, local_labels in generator:  # 获取输入数据

            truth_orient_patch = local_labels["Orientation_patch"].float(
            ).cuda()
            truth_conf_patch = local_labels["Confidence_patch"].long().cuda()
            truth_orient_yaw = local_labels["Orientation_yaw"].float().cuda()
            truth_conf_yaw = local_labels["Confidence_yaw"].long().cuda()

            local_batch = local_batch.float().cuda()
            [orient_patch, conf_patch, orient_yaw,
             conf_yaw] = model(local_batch)

            orient_patch_loss = orient_loss_func(orient_patch,
                                                 truth_orient_patch,
                                                 truth_conf_patch)

            # softmax函数的输出值进行操作，每行——>返回每行最大值的索引
            truth_conf_patch = torch.max(truth_conf_patch, dim=1)[1]
            conf_patch_loss = conf_loss_func(conf_patch, truth_conf_patch)

            loss_patch = conf_patch_loss + w * orient_patch_loss

            orient_yaw_loss = orient_loss_func(orient_yaw, truth_orient_yaw,
                                               truth_conf_yaw)

            # softmax函数的输出值进行操作，每行——>返回每行最大值的索引
            truth_conf_yaw = torch.max(truth_conf_yaw, dim=1)[1]
            conf_yaw_loss = conf_loss_func(conf_yaw, truth_conf_yaw)

            loss_yaw = conf_yaw_loss + w * orient_yaw_loss

            total_loss = alpha * loss_patch + loss_yaw

            opt_RMSprop.zero_grad()  # 梯度置0
            total_loss.backward()  # 反向传播
            opt_RMSprop.step()  # 优化

            if passes % 10 == 0:  # 10轮显示一次，打印状态信息
                print(
                    "--- epoch %s | batch %s/%s --- [loss_patch: %s] | [loss_yaw: %s] | [total_loss: %s]"
                    % (
                        epoch,
                        curr_batch,
                        total_num_batches,
                        loss_patch.item(),
                        loss_yaw.item(),
                        total_loss.item(),
                    ))
                passes = 0

            passes += 1
            curr_batch += 1
        writer.add_scalar("loss_total", total_loss, epoch)
        writer.add_scalar("loss_patch", loss_patch, epoch)
        writer.add_scalar("orient_patch_loss", orient_patch_loss, epoch)
        writer.add_scalar("conf_patch_loss", conf_patch_loss, epoch)

        # save after every 10 epochs
        if epoch % 20 == 0:
            name = model_path + "epoch_%s.pkl" % epoch
            print("====================")
            print("Done with epoch %s!" % epoch)
            print("Saving weights as %s ..." % name)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt_SGD.state_dict(),
                    "loss": total_loss,
                },
                name,
            )
            print("====================")

    writer.close()


if __name__ == "__main__":
    main()