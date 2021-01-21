from torch_lib.dataset_posenet import *
from contrast_experiment.model_six import Model


import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg, resnet, densenet, mobilenet
from torch.utils import data

from tensorboardX import SummaryWriter

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():

    # hyper parameters
    epochs = 250
    batch_size = 8
    w = 1
    alpha = 1

    print("Loading all detected objects in dataset...")

    train_path = "/media/lab/TOSHIBAEXT/BuildingData/training"
    dataset = Dataset(train_path)  # 自定义的数据集

    params = {"batch_size": batch_size, "shuffle": True, "num_workers": 6}

    generator = data.DataLoader(dataset, **params)  # 读取Dataset中的数据

    my_vgg = mobilenet.mobilenet_v2(pretrained=True)  # 加载模型并设置为预训练模式
    model = Model(features=my_vgg, fixed_weight=True).cuda()
    # my_vgg = resnet.resnet34(pretrained=True)  # 加载模型并设置为预训练模式
    # model = HourglassNet(base_model=my_vgg).cuda()

    opt_SGD = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    # conf_loss_func = nn.CrossEntropyLoss().cuda()
    rotation_loss = torch.nn.MSELoss().cuda()

    # load any previous weights
    model_path = '/home/lab/Desktop/wzndeep/posenet-build--eular/contrast_experiment/weights/'
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
        '/home/lab/Desktop/wzndeep/posenet-build--eular/contrast_experiment/runs')

    for epoch in range(first_epoch + 1, epochs + 1):  # 多批次循环
        curr_batch = 0
        passes = 0
        for local_batch, local_labels in generator:  # 获取输入数据

            truth_eular = local_labels["Eular_six"].float().cuda()

            local_batch = local_batch.float().cuda()
            out_eular = model(local_batch)

            loss = rotation_loss(out_eular, truth_eular)

            opt_SGD.zero_grad()  # 梯度置0
            loss.backward()  # 反向传播
            opt_SGD.step()  # 优化

            if passes % 10 == 0:  # 10轮显示一次，打印状态信息
                print(
                    "---{} ---Loss: total loss {:.3f}".format(
                        epoch, loss
                    )
                )
                passes = 0

            passes += 1
            curr_batch += 1
        writer.add_scalar("loss_total", loss, epoch)

        # save after every 10 epochs
        if epoch % 10 == 0:
            name = model_path + "epoch_%s.pkl" % epoch
            print("====================")
            print("Done with epoch %s!" % epoch)
            print("Saving weights as %s ..." % name)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt_SGD.state_dict(),
                    "loss": loss,
                },
                name,
            )
            print("====================")

    writer.close()


if __name__ == "__main__":
    main()
