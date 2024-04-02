#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
tpaddle tool 5
"""
import argparse
import paddle
import paddle.nn.functional as F
import paddle.vision.models as models
from paddle.io import Dataset
from paddle.vision.transforms import Compose, Resize, ToTensor
import numpy as np


parser = argparse.ArgumentParser(__doc__)
parser.add_argument(
    "--model_name", type=str, default='alexnet', help="model name for GPU train."
)
args = parser.parse_args()

models_zoo = {'mobilenetv1': models.MobileNetV1,
              'mobilenetv2': models.MobileNetV2,
              'resnet18': models.resnet18,
              'resnet34': models.resnet34,
              'resnet50': models.resnet50,
              'resnet101': models.resnet101,
              'vgg11': models.vgg11,
              'vgg13': models.vgg13,
              'vgg16': models.vgg16,
              'vgg19': models.vgg19}


class MyDataset(Dataset):
    def __init__(self, pre_dataset, num_samples):
        self.num_samples = num_samples
        self.pre_dataset = pre_dataset

    def __getitem__(self, idx):
        image = self.pre_dataset[idx][0]
        label = self.pre_dataset[idx][1]
        return image, label

    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    paddle.device.set_device('cpu')
    # 加载数据集
    transform = Compose([Resize(size=224), ToTensor()])
    cifar10_train = paddle.vision.datasets.Cifar10(mode='train',
                                                   transform=transform)
    cifar10_test = paddle.vision.datasets.Cifar10(mode='test',
                                                  transform=transform)

    cifar10_train = MyDataset(cifar10_train, 400)
    cifar10_test = MyDataset(cifar10_test, 80)

    # model = paddle.vision.models.MobileNetV2()

    model = models_zoo[args.model_name]
    model = model()
    epoch_num = 1
    batch_size = 2
    learning_rate = 0.001

    val_acc_history = []
    val_loss_history = []


    def train(model):
        print('start training ... ')
        # turn into training mode
        model.train()

        opt = paddle.optimizer.Adam(learning_rate=learning_rate,
                                    parameters=model.parameters())

        train_loader = paddle.io.DataLoader(cifar10_train,
                                            shuffle=True,
                                            batch_size=batch_size)

        valid_loader = paddle.io.DataLoader(cifar10_test, batch_size=batch_size)

        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_loader()):
                x_data = data[0]
                y_data = paddle.to_tensor(data[1])
                y_data = paddle.unsqueeze(y_data, 1)

                logits = model(x_data)
                loss = F.cross_entropy(logits, y_data)

                if batch_id % 10 == 0:
                    print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))
                loss.backward()
                opt.step()
                opt.clear_grad()

            # evaluate model after one epoch
            model.eval()
            accuracies = []
            losses = []
            for batch_id, data in enumerate(valid_loader()):
                x_data = data[0]
                y_data = paddle.to_tensor(data[1])
                y_data = paddle.unsqueeze(y_data, 1)

                logits = model(x_data)
                loss = F.cross_entropy(logits, y_data)
                acc = paddle.metric.accuracy(logits, y_data)
                accuracies.append(acc.numpy())
                losses.append(loss.numpy())

            avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
            print("[validation] accuracy/loss: {}/{}".format(avg_acc, avg_loss))
            val_acc_history.append(avg_acc)
            val_loss_history.append(avg_loss)
            model.train()


    train(model)
