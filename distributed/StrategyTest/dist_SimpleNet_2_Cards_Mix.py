#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
TP 策略 使用行分割 非分布式输入
"""
import paddle
import numpy as np
import random
import paddle.distributed as dist
import paddle.nn.functional as F
import paddle.distributed.fleet as fleet
def set_random_seed(seed):
   random.seed(seed)
   np.random.seed(seed)
   paddle.seed(seed)
set_random_seed(33)
class SimpleMPNet(paddle.nn.Layer):
   def __init__(self):
      super().__init__()
      self.flatten = paddle.nn.Flatten(1, -1)
      self.linear1 = fleet.meta_parallel.ColumnParallelLinear(
            784,
            392,
            gather_output=False,
            has_bias=True)
      self.linear2 = fleet.meta_parallel.RowParallelLinear(
            392,
            128,
            input_is_parallel=True,
            has_bias=True)
      self.linear3 = paddle.nn.Linear(128, 10)


   def forward(self, x):
      x = self.flatten(x)
      x = self.linear1(x)
      x = self.linear2(x)
      x = F.relu(x)
      x = F.dropout(x, 0.2)
      # x = self.linear2(x)
      x = self.linear3(x)
      return x


# 1、初始化分布式环境

strategy = fleet.DistributedStrategy()

# 设置两路张量模型并行
model_parallel_size = 2
data_parallel_size = 1
strategy.hybrid_configs = {
   "dp_degree": data_parallel_size,
   "mp_degree": model_parallel_size,
   "pp_degree": 1
}
# 注意 strategy 是这里传递的，动态图只能这里，静态图还可以在 distributed_optimizer 里传
fleet.init(is_collective=True, strategy=strategy)


model = SimpleMPNet()
model = fleet.distributed_model(model)

# 设置优化器
optimizer = paddle.optimizer.Adam(parameters=model.parameters(),learning_rate=0.002)
optimizer = fleet.distributed_optimizer(optimizer)
from paddle.vision.datasets import MNIST
from paddle.vision.transforms import ToTensor
import numpy as np
import paddle.nn.functional as F
from paddle.distributed import fleet
from paddle.io import DataLoader, DistributedBatchSampler
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
        optimizer.step()
        optimizer.clear_grad()
print("acc: {}".format(acc))

