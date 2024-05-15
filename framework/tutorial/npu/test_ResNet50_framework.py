#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_ResNet50_framework.py
"""
import random
import paddle
from paddle.vision import transforms
from paddle.vision.models import resnet50
import numpy as np

# 1. 设定运行设备为npu
paddle.set_device("npu")


def set_random_seed(seed):
    """
    随机数种子设置
    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


set_random_seed(33)
max_iter = 10

# 2. 定义数据集、数据预处理方法与 DataLoader
transform = transforms.Compose(
    [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
train_set = paddle.vision.datasets.Cifar10(mode="train", transform=transform)
train_loader = paddle.io.DataLoader(train_set, batch_size=256, num_workers=8)

# 3. 定义网络结构
net = resnet50(num_classes=10)
# 4. 定义损失函数
net_loss = paddle.nn.CrossEntropyLoss()
# 5. 定义优化器
optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

net.train()
for epoch in range(1):
    for batch_idx, data in enumerate(train_loader, start=0):
        inputs, labels = data
        optimizer.clear_grad()
        # 6. 前向传播并计算损失
        outputs = net(inputs)
        loss = net_loss(outputs, labels)
        # 7. 反向传播
        loss.backward()
        # 8. 更新参数
        optimizer.step()
        print("Epoch %d, Iter %d, Loss: %.5f" % (epoch + 1, batch_idx + 1, loss))
        if batch_idx + 1 == max_iter:
            break
print("Finished Training")

test_dataset = paddle.vision.datasets.Cifar10(mode="test", transform=transform)

# 测试5张图片效果
for i in range(5):
    test_image, gt = test_dataset[0]
    # CHW -> NCHW
    test_image = test_image.unsqueeze(0)

    # 取预测分布中的最大值
    res = net(test_image).argmax().numpy()
    print(f"图像{i} 标签：{gt}")
    print(f"模型预测结果：{res}")
