#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
PP流水线并行
"""
import numpy as np
import os
import paddle
from paddle.distributed import fleet
from paddle.nn import Sequential
import paddle.nn as nn
from paddle.nn import Layer
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer
import paddle.nn.functional as F
import paddle.distributed as dist
import random
from paddle.io import Dataset, BatchSampler, DataLoader
from paddle.io import DataLoader, DistributedBatchSampler
from paddle.vision.datasets import MNIST
from paddle.vision.transforms import ToTensor



strategy = fleet.DistributedStrategy()
model_parallel_size = 1
data_parallel_size = 1
pipeline_parallel_size = 6
strategy.hybrid_configs = {
    "dp_degree": data_parallel_size,
    "mp_degree": model_parallel_size,
    "pp_degree": pipeline_parallel_size
}
strategy.pipeline_configs = {
    "accumulate_steps": 8,
    "micro_batch_size": 8
}

fleet.init(is_collective=True, strategy=strategy)



def set_random_seed(seed, dp_id, rank_id):
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + dp_id + rank_id)
    print("seed: ", seed)
    print("rank_id: ", rank_id)
    print("dp_id: ", dp_id)

hcg = fleet.get_hybrid_communicate_group()
world_size = hcg.get_model_parallel_world_size()
dp_id = hcg.get_data_parallel_rank()
pp_id = hcg.get_stage_id()
rank_id = dist.get_rank()
set_random_seed(1024, dp_id, rank_id)


class ReshapeHelp(Layer):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(shape=self.shape)


class SimpleNetPipeDesc(PipelineLayer):
    def __init__(self, num_classes=10, **kwargs):
        self.num_classes = num_classes
        decs = [
            LayerDesc(
                nn.Flatten, 1, -1),
            LayerDesc(
                nn.Linear, 784, 512),
            LayerDesc(nn.ReLU),
            LayerDesc(
                nn.Dropout, 0.2),
            LayerDesc(
                nn.Linear, 512, 10),
            LayerDesc(nn.ReLU),
            LayerDesc(
                nn.Dropout, 0.2),
            LayerDesc(
                nn.Linear, 10, 10),
            LayerDesc(nn.ReLU),
            LayerDesc(
                nn.Dropout, 0.2),
            LayerDesc(
                nn.Linear, 10, 10),
            LayerDesc(nn.ReLU),
            LayerDesc(
                nn.Dropout, 0.2),
            LayerDesc(
                nn.Linear, 10, 10),
            LayerDesc(nn.ReLU),
            LayerDesc(
                nn.Dropout, 0.2),
            LayerDesc(
                nn.Linear, 10, 10),
            LayerDesc(nn.ReLU),
            LayerDesc(
                nn.Dropout, 0.2),
            LayerDesc(
                nn.Linear, 10, 10),
            LayerDesc(nn.ReLU),
            LayerDesc(
                nn.Dropout, 0.2),
            LayerDesc(
                nn.Linear, 10, 10),
            LayerDesc(nn.ReLU),
            LayerDesc(
                nn.Dropout, 0.2),
            LayerDesc(
                nn.Linear, 10, 10),
            LayerDesc(nn.ReLU),
            LayerDesc(
                nn.Dropout, 0.2),
            LayerDesc(
                nn.Linear, 10, 10),
        ]
        super().__init__(
            layers=decs, loss_fn=nn.CrossEntropyLoss(), **kwargs)

model = SimpleNetPipeDesc(num_stages=pipeline_parallel_size, topology=hcg._topo)
opt = paddle.optimizer.Adam(parameters=model.parameters(),learning_rate=0.002)
model = fleet.distributed_model(model)
opt = fleet.distributed_optimizer(opt)

train_dataset = MNIST(mode='train', transform=ToTensor())
test_dataset = MNIST(mode='test', transform=ToTensor())
# 五、构建分布式训练使用的数据集
train_sampler = DistributedBatchSampler(train_dataset, 64, shuffle=False, drop_last=True)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=2)

for epoch in range(1):
    model.train()
    for batch_id, data in enumerate(train_loader()):
        x_data = data[0]
        y_data = data[1]  # 训练数据标签
        loss = model.train_batch([x_data, y_data], opt)
        loss.backward()
        if batch_id % 100 == 0:
            print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))
        opt.step()
        opt.clear_grad()
