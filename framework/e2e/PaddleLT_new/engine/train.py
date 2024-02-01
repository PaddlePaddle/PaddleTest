#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
train 方法
"""
import numpy as np
import paddle
from engine.xtools import reset

from generator.builder_layer import BuildLayer
from generator.builder_data import BuildData
from generator.builder_optimizer import BuildOptimizer
from generator.builder_loss import BuildLoss


class LayerTrain(object):
    """
    构建Layer训练的通用类
    """

    # def __init__(self, testing, case, layer):
    def __init__(self, testing, layerfile):
        """
        初始化
        """
        self.seed = 33
        reset(self.seed)

        self.testing = testing

        self.model_dtype = self.testing.get("model_dtype")
        paddle.set_default_dtype(self.model_dtype)

        self.net = BuildLayer(layerfile=layerfile).get_layer()

        self.data = BuildData(layerfile=layerfile).get_single_data()

        # self.optimizer = optimizer
        self.optimizer_name = self.testing.get("optimizer").get("optimizer_name")
        self.optimizer_param = self.testing.get("optimizer").get("params")
        self.optimizer = BuildOptimizer(optimizer_name=self.optimizer_name, optimizer_param=self.optimizer_param)

        # self.loss = loss
        self.loss_name = self.testing.get("Loss").get("loss_name")
        self.loss_param = self.testing.get("Loss").get("params")
        self.loss = BuildLoss(loss_name=self.loss_name, loss_param=self.loss_param)

        self.step = self.testing.get("step")

    def dy_train(self):
        """dygraph train"""

        if not self.net.parameters():
            return "pass"

        reset(self.seed)

        # net = self.net.get_layer()
        self.net.train()
        # print(self.net.parameters()) 打印参数parameters

        # 构建optimizer用于训练
        opt = self.optimizer.get_opt(net=self.net)

        for epoch in range(self.step):
            logit = self.net(*self.data)
            # 构建loss用于训练
            loss = self.loss.get_loss(logit)
            loss.backward()
            opt.step()
            opt.clear_grad()
        return logit

    # def dy_train_dl(self):
    #     """dygraph train with dataloader"""
    #     reset(self.seed)

    #     # net = self.net.get_layer()
    #     self.net.train()

    #     # 构建optimizer用于训练
    #     opt = self.optimizer.get_opt(net=self.net)

    #     for epoch in range(self.step):
    #         for i, data_dict in enumerate(self.data()):
    #             logit = self.net(**data_dict)
    #             # 构建loss用于训练
    #             # logit = self.loss_info.get_loss(logit)
    #             loss = self.loss.get_loss(logit)
    #             loss.backward()
    #             opt.step()
    #             opt.clear_grad()
    #     return logit

    def dy2st_train(self):
        """dy2st train"""

        if not self.net.parameters():
            return "pass"

        reset(self.seed)

        # net = self.net.get_layer()
        net = paddle.jit.to_static(self.net, full_graph=True)
        net.train()
        # 构建optimizer用于训练
        opt = self.optimizer.get_opt(net=net)

        for epoch in range(self.step):
            # data_dict = self.data[epoch]
            logit = net(*self.data)
            # 构建loss用于训练
            loss = self.loss.get_loss(logit)
            loss.backward()
            opt.step()
            opt.clear_grad()
        return logit

    def dy2st_train_cinn(self):
        """dy2st train"""

        if not self.net.parameters():
            return "pass"

        reset(self.seed)

        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = True
        net = paddle.jit.to_static(self.net, build_strategy=build_strategy, full_graph=True)
        net.train()
        # 构建optimizer用于训练
        opt = self.optimizer.get_opt(net=net)

        for epoch in range(self.step):
            logit = net(*self.data)
            # 构建loss用于训练
            loss = self.loss.get_loss(logit)
            loss.backward()
            opt.step()
            opt.clear_grad()
        return logit
