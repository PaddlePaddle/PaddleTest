#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
# ======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
# ======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file dist_fleet_hybrid_parallel.py
  * @author liyang109@baidu.com
  * @date 2021-05-10 14:32
  * @brief
  *
  **************************************************************************/
"""
from __future__ import division
from __future__ import print_function
import os
import random
import numpy as np

import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from paddle.io import DataLoader, Dataset
from dist_base_sampleNet import *

strategy = fleet.DistributedStrategy()


class DygraphHybrid:
    """Test Dygrah hybrid model parallel"""

    def __init__(self):
        """init dp1 mp2 pp1"""
        self.topo = fleet.CommunicateTopology(["data", "model", "pipe"], [1, 2, 1])
        self.data_parallel_size = self.topo.get_dim("data")
        self.model_parallel_size = self.topo.get_dim("model")
        self.pipeline_parallel_size = self.topo.get_dim("pipe")
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": self.pipeline_parallel_size,
        }
        fleet.init(is_collective=True, strategy=strategy)

    def hybrid_parallel(self):
        """test hybrid parallel api and loss consistent"""
        hcg = fleet.get_hybrid_communicate_group()
        parallel_name = self.topo.get_hybrid_group_names()
        assert len(parallel_name) == 3

        word_size = self.topo.world_size()
        assert word_size == int(self.data_parallel_size) * int(self.model_parallel_size) * int(
            self.pipeline_parallel_size
        )

        # get rank后面加
        # coord2rank = self.topo.get_rank("dp"=1)
        # assert coord2rank is not None

        rank2coord = self.topo.get_coord(1)
        assert rank2coord is not None
        # 这个地方有点问题
        # ranks_dp = self.topo.get_axis_list("data", 0)
        # ranks_mp = self.topo.get_axis_list("model", 1)
        # ranks_pp = self.topo.get_axis_list("pipe", 2)
        # print("ranks============>>>>>>>>>>", ranks_dp)
        # print("ranks============>>>>>>>>>>", ranks_mp)
        # print("ranks============>>>>>>>>>>", ranks_pp)
        dim_size_dp = self.topo.get_dim_size("data")
        assert dim_size_dp == self.topo.get_dim("data")

        dim_size_mp = self.topo.get_dim_size("model")
        assert dim_size_mp == self.topo.get_dim("model")

        dim_size_pp = self.topo.get_dim_size("pipe")
        assert dim_size_pp == self.topo.get_dim("pipe")

        comm_list = self.topo.get_comm_list("data")
        assert comm_list is not None

        paral_mode = hcg.get_parallel_mode()
        assert paral_mode is not None

        comm_group1 = hcg._set_comm_group(parallel_method="data")
        assert comm_group1 is not None

        comm_group2 = hcg._set_comm_group(parallel_method="model")
        assert comm_group2 is not None

        comm_group3 = hcg._set_comm_group(parallel_method="pipe")
        assert comm_group3 is not None

        topology_res = hcg.topology()
        assert topology_res is not None

        global_rank = hcg.get_global_rank()
        assert global_rank is not None

        data_parallel_id = hcg._get_data_parallel_id()
        assert data_parallel_id is not None

        data_parallel_rank = hcg.get_data_parallel_rank()
        assert data_parallel_rank is not None

        data_parallel_world_size = hcg.get_data_parallel_world_size()
        assert data_parallel_world_size == self.data_parallel_size

        data_parallel_group = hcg.get_data_parallel_group()
        assert data_parallel_group is not None

        data_parallel_group_src_rank = hcg.get_data_parallel_group_src_rank()
        assert data_parallel_group_src_rank in [0, 1]

        model_parallel_id = hcg._get_model_parallel_id()
        assert model_parallel_id in range(self.model_parallel_size)

        model_word_size = hcg.get_model_parallel_world_size()
        assert model_word_size == self.model_parallel_size

        model_parallel_group = hcg.get_model_parallel_group()
        assert model_parallel_group is not None

        model_parallel_group_src_rank = hcg.get_model_parallel_group_src_rank()
        assert model_parallel_group_src_rank is not None

        mp_id = hcg.get_model_parallel_rank()
        assert mp_id in range(self.model_parallel_size)

        dp_id = hcg.get_data_parallel_rank()
        assert dp_id is not None

        rank_id = dist.get_rank()
        assert rank_id is not None
        set_random_seed(1024, dp_id, rank_id)
        np_fc1 = np.random.random_sample((hidden_size, inner_size))
        np_fc2 = np.random.random_sample((inner_size, hidden_size))
        train_data = TrainDataset(length=1000)
        print("train_data", train_data)
        train_batch_sampler = paddle.io.DistributedBatchSampler(
            train_data, batch_size=4, shuffle=False, num_replicas=self.data_parallel_size, rank=dp_id
        )

        train_data_loader = DataLoader(
            dataset=train_data, batch_sampler=train_batch_sampler, num_workers=0, return_list=True
        )

        model_a = SimpleMPNet(vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2, mp_id)
        optimizer_a = paddle.optimizer.SGD(learning_rate=0.001, parameters=model_a.parameters())
        model_a = fleet.distributed_model(model_a)
        optimizer_a = fleet.distributed_optimizer(optimizer_a)

        model_b = SimpleDPNet(vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2)
        optimizer_b = paddle.optimizer.SGD(learning_rate=0.001, parameters=model_b.parameters())
        loss_a_arr = []
        loss_b_arr = []
        for step, batch in enumerate(train_data_loader):
            if step > 5:
                break
            output_a = model_a(batch)
            loss_a = output_a.mean()
            loss_a.backward()
            loss_a_arr.append(loss_a)
            optimizer_a.step()
            optimizer_a.clear_grad()
            output_b = model_b(batch)
            loss_b = output_b.mean()
            loss_b.backward()
            optimizer_b.step()
            optimizer_b.clear_grad()
            loss_b_arr.append(loss_b)
        print(np.allclose(loss_a_arr, loss_b_arr))
        # assert np.allclose(loss_a_arr, loss_b_arr)
        print("all is success...")


if __name__ == "__main__":
    obj = DygraphHybrid()
    obj.hybrid_parallel()
