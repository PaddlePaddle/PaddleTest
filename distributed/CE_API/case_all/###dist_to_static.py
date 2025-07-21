#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
# ======================================================================
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
# ======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2025 Baidu.com, Inc. All Rights Reserved
  * @file dist_to_static.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-08
  * @brief
  *
  **************************************************************************/
"""
import numpy as np
import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import Replicate, Shard
from utils import run_priority

BATCH_SIZE = 4
BATCH_NUM = 4
IMAGE_SIZE = 16
CLASS_NUM = 8

class RandomDataset(paddle.io.Dataset):
    """RandomDataset"""
    def __init__(self, images, labels, num_samples):
        """__init__"""
        self.images = images
        self.labels = labels
        self.num_samples = num_samples
    def __getitem__(self, idx):
        """__getitem__"""
        return self.images[idx], self.labels[idx]
    def __len__(self):
        """__len__"""
        return self.num_samples

class DemoNet(nn.Layer):
    """DemoNet"""
    def __init__(self, mesh):
        """__init__"""
        super().__init__()
        self._mesh = mesh
        self.linear_0 = nn.Linear(IMAGE_SIZE, IMAGE_SIZE)
        self.linear_1 = nn.Linear(IMAGE_SIZE, CLASS_NUM)
        self.relu = nn.ReLU()
        # shard the weights of this layer
        self.linear_0.weight = dist.shard_tensor(
            self.linear_0.weight,
            self._mesh,
            [Shard(1)],
            stop_gradient=False,
        )
        self.linear_1.weight = dist.shard_tensor(
            self.linear_1.weight,
            self._mesh,
            [Shard(0)],
            stop_gradient=False,
        )
    def forward(self, x):
        """forward"""
        out = self.linear_0(x)
        out = self.relu(out)
        out = self.linear_1(out)
        return out

@run_priority(level="P0")
def test_to_static():
    """test_to_static"""
    images = np.random.rand(BATCH_SIZE, IMAGE_SIZE).astype('float32')
    labels = np.random.rand(BATCH_SIZE, CLASS_NUM).astype('float32')
    dataset = RandomDataset(images, labels, BATCH_SIZE)
    loader = paddle.io.DataLoader(dataset, batch_size=BATCH_SIZE)

    mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
    layer = DemoNet(mesh)
    opt = paddle.optimizer.SGD(
        learning_rate=0.1, parameters=layer.parameters()
    )
    loss_fn = nn.MSELoss()
    dist_loader = dist.shard_dataloader(loader, meshes=[mesh])
    dist_model = dist.to_static(
        layer, dist_loader, loss_fn, opt
    )
    # training
    dist_model.train()
    for batch_id, (image, label) in enumerate(dist_loader()):
        loss = dist_model(image, label)
        assert loss.ndim == 0


    # evaluation
    dist_model.eval()
    for batch_id, (image, label) in enumerate(dist_loader()):
        loss = dist_model(image, label)
        assert loss.ndim == 0

    # prediction
    dist_model.predict()
    for batch_id, (image, label) in enumerate(dist_loader()):
        outs = dist_model(image)    
        assert isinstance(outs, dict)
        output_tensor = outs['out0']
        assert output_tensor.shape == (BATCH_SIZE, CLASS_NUM)

    print("test_to_static ... ok")


if __name__ == "__main__":
    test_to_static()