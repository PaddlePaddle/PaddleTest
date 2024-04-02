#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
mnist
"""
import paddle
from paddle.vision.datasets import MNIST
from paddle.vision.transforms import ToTensor
import numpy as np
import paddle.nn.functional as F
from paddle.distributed import fleet
from paddle.io import DataLoader, DistributedBatchSampler

paddle.seed(33)

print(paddle.__version__)
# 设置 GPU 环境
paddle.set_device('gpu')
np.random.seed(33)

paddle.seed(33)
# 模型组网，构建并初始化一个模型 mnist
mnist = paddle.nn.Sequential(
    paddle.nn.Flatten(1, -1),
    paddle.nn.Linear(784, 512),
    paddle.nn.ReLU(),
    paddle.nn.Dropout(0.2),
    paddle.nn.Linear(512, 10)
)

# 将mnist模型及其所有子层设置为训练模式。这只会影响某些模块，如Dropout和BatchNorm。

def train():
    # 二、初始化 Fleet 环境
    fleet.init(is_collective=True)
    # 三、构建分布式训练使用的网络模型
    model = fleet.distributed_model(mnist)

    # 设置优化器
    opt = paddle.optimizer.Adam(parameters=mnist.parameters(),learning_rate=0.002)
    # 四、构建分布式训练使用的优化器
    opt = fleet.distributed_optimizer(opt)

    train_dataset = MNIST(mode='train', transform=ToTensor())
    test_dataset = MNIST(mode='test', transform=ToTensor())
    # 五、构建分布式训练使用的数据集
    train_sampler = DistributedBatchSampler(train_dataset, 32, shuffle=False, drop_last=False)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=2)

    loss_fn = paddle.nn.CrossEntropyLoss()
    acc = 0
    for epoch in range(1):
        model.train()
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]  # 训练数据标签

            predicts = model(x_data)
            loss = loss_fn(predicts, y_data)
            # 计算准确率 等价于 prepare 中metrics的设置
            acc = paddle.metric.accuracy(predicts, y_data)
            loss.backward()
            if batch_id % 100 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))
            opt.step()
            opt.clear_grad()
    print("acc: {}".format(acc))


if __name__ == "__main__":
    train()
