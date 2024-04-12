#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
train 方法
"""
import os
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

    # def __init__(self, testing, layerfile, device_id):
    def __init__(self, testing, layerfile):
        """
        初始化
        """
        self.seed = 33
        reset(self.seed)
        self.device = os.environ.get("PLT_SET_DEVICE")
        paddle.set_device(str(self.device))
        # paddle.set_device("{}:{}".format(str(self.device), str(device_id)))

        self.testing = testing
        self.model_dtype = self.testing.get("model_dtype")
        paddle.set_default_dtype(self.model_dtype)

        self.layerfile = layerfile
        self.step = self.testing.get("step")

        # self.net = BuildLayer(layerfile=layerfile).get_layer()

        # self.data = BuildData(layerfile=layerfile).get_single_data()

        # self.optimizer_name = self.testing.get("optimizer").get("optimizer_name")
        # self.optimizer_param = self.testing.get("optimizer").get("params")
        # self.optimizer = BuildOptimizer(optimizer_name=self.optimizer_name, optimizer_param=self.optimizer_param)

        # self.loss_name = self.testing.get("Loss").get("loss_name")
        # self.loss_param = self.testing.get("Loss").get("params")
        # self.loss = BuildLoss(loss_name=self.loss_name, loss_param=self.loss_param)

    def _get_instant(self):
        """get data, net, optimizer, loss"""
        reset(self.seed)

        data = BuildData(layerfile=self.layerfile).get_single_data()
        net = BuildLayer(layerfile=self.layerfile).get_layer()

        optimizer_name = self.testing.get("optimizer").get("optimizer_name")
        optimizer_param = self.testing.get("optimizer").get("params")
        optimizer = BuildOptimizer(optimizer_name=optimizer_name, optimizer_param=optimizer_param)

        loss_name = self.testing.get("Loss").get("loss_name")
        loss_param = self.testing.get("Loss").get("params")
        loss = BuildLoss(loss_name=loss_name, loss_param=loss_param)
        return data, net, optimizer, loss

    def _get_data_grad(self, data):
        """记录list[inputs...]中的input.grad并生成list[input.grad...]"""
        data_grad = []
        for i in data:
            data_grad.append(i.grad)
        return data_grad

    def dy_train(self):
        """dygraph train"""

        # if not self.net.parameters():
        #     return "pass"

        data, net, optimizer, loss = self._get_instant()

        net.train()
        # print(self.net.parameters()) 打印参数parameters

        # 构建optimizer用于训练
        if net.parameters():
            opt = optimizer.get_opt(net=net)

        for epoch in range(self.step):
            logit = net(*data)
            # 构建loss用于训练
            dy_loss = loss.get_loss(logit)
            dy_loss.backward()
            if net.parameters():
                opt.step()
                opt.clear_grad()

        data_grad = self._get_data_grad(data)
        return {"logit": logit, "data_grad": data_grad}

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

        data, net, optimizer, loss = self._get_instant()

        net.train()
        st_net = paddle.jit.to_static(net, full_graph=True)

        # 构建optimizer用于训练
        if st_net.parameters():
            opt = optimizer.get_opt(net=st_net)

        for epoch in range(self.step):
            logit = st_net(*data)
            # 构建loss用于训练
            dy_loss = loss.get_loss(logit)
            dy_loss.backward()
            if st_net.parameters():
                opt.step()
                opt.clear_grad()

        data_grad = self._get_data_grad(data)
        return {"logit": logit, "data_grad": data_grad}

    def dy2st_train_cinn(self):
        """dy2st train"""

        # if not self.net.parameters():
        #     return "pass"

        data, net, optimizer, loss = self._get_instant()

        net.train()
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = True
        st_net = paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True)

        # 构建optimizer用于训练
        if st_net.parameters():
            opt = optimizer.get_opt(net=st_net)

        for epoch in range(self.step):
            logit = st_net(*data)
            # 构建loss用于训练
            dy_loss = loss.get_loss(logit)
            dy_loss.backward()
            if st_net.parameters():
                opt.step()
                opt.clear_grad()

        data_grad = self._get_data_grad(data)
        return {"logit": logit, "data_grad": data_grad}
