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

    def __init__(self, testing, case, layer):
        """
        初始化
        """
        self.seed = 33
        reset(self.seed)

        self.testing = testing
        self.layer = layer
        self.case = case

        self.layer_name = self.layer.get("Layer").get("layer_name")
        self.layer_param = self.layer.get("Layer").get("params")
        self.net = BuildLayer(layer_name=self.layer_name, layer_param=self.layer_param)

        self.data_info = self.layer.get("DataGenerator")
        self.data = BuildData(data_info=self.data_info).get_single_data()

        # self.optimizer = optimizer
        self.optimizer_name = self.testing.get("optimizer").get("optimizer_name")
        self.optimizer_param = self.testing.get("optimizer").get("params")
        self.optimizer = BuildOptimizer(optimizer_name=self.optimizer_name, optimizer_param=self.optimizer_param)

        # self.loss = loss
        self.loss_name = self.testing.get("Loss").get("loss_name")
        self.loss_param = self.testing.get("Loss").get("params")
        self.loss = BuildLoss(loss_name=self.loss_name, loss_param=self.loss_param)

        self.step = self.testing.get("step")

        self.model_dtype = self.testing.get("model_dtype")
        paddle.set_default_dtype(self.model_dtype)

    def dy_train(self):
        """dygraph train"""
        reset(self.seed)

        net = self.net.get_layer()
        net.train()

        # 构建optimizer用于训练
        opt = self.optimizer.get_opt(net=net)

        for epoch in range(self.step):
            data_dict = self.data[epoch]
            logit = net(**data_dict)
            # 构建loss用于训练
            loss = self.loss.get_loss(logit)
            loss.backward()
            opt.step()
            opt.clear_grad()
        return logit

    def dy_train_dl(self):
        """dygraph train with dataloader"""
        reset(self.seed)

        # net = self.layer_info.get_layer()
        net = self.net.get_layer()
        net.train()
        # 构建optimizer用于训练
        opt = self.optimizer.get_opt(net=net)

        for epoch in range(self.step):
            for i, data_dict in enumerate(self.data()):
                logit = net(**data_dict)
                # 构建loss用于训练
                # logit = self.loss_info.get_loss(logit)
                loss = self.loss.get_loss(logit)
                loss.backward()
                opt.step()
                opt.clear_grad()
        return logit

    def dy2st_train(self):
        """dy2st train"""
        reset(self.seed)

        # net = self.layer_info.get_layer()
        net = self.net.get_layer()
        net = paddle.jit.to_static(net)
        net.train()
        # 构建optimizer用于训练
        opt = self.optimizer.get_opt(net=net)

        for epoch in range(self.step):
            # data_module_type == 'Dataset'
            data_dict = self.data[epoch]
            logit = net(**data_dict)
            # 构建loss用于训练
            # logit = self.loss_info.get_loss(logit)
            loss = self.loss.get_loss(logit)
            loss.backward()
            opt.step()
            opt.clear_grad()
        return logit

    def dy2st_train_dl(self):
        """dy2st train with dataloader"""
        reset(self.seed)

        # net = self.layer_info.get_layer()
        net = self.net.get_layer()
        net = paddle.jit.to_static(net)
        net.train()
        # 构建optimizer用于训练
        opt = self.optimizer.get_opt(net=net)

        for epoch in range(self.step):
            for i, data_dict in enumerate(self.data()):
                logit = net(**data_dict)
                # 构建loss用于训练
                # logit = self.loss_info.get_loss(logit)
                loss = self.loss.get_loss(logit)
                loss.backward()
                opt.step()
                opt.clear_grad()
        return logit

    def dy2st_train_cinn(self):
        """dy2st train"""
        reset(self.seed)

        # net = self.layer_info.get_layer()
        net = self.net.get_layer()
        net = paddle.jit.to_static(net, backend="CINN")
        net.train()
        # 构建optimizer用于训练
        opt = self.optimizer.get_opt(net=net)

        for epoch in range(self.step):
            # data_module_type == 'Dataset'
            data_dict = self.data[epoch]
            logit = net(**data_dict)
            # 构建loss用于训练
            # logit = self.loss_info.get_loss(logit)
            loss = self.loss.get_loss(logit)
            loss.backward()
            opt.step()
            opt.clear_grad()
        return logit

    # def dy_train_dl(self, to_static=False):
    #     """dygraph or static train"""
    #     paddle.enable_static()
    #     paddle.disable_static()
    #     paddle.seed(33)
    #     np.random.seed(33)
    #     net = self.layer_info.get_layer()
    #
    #     if to_static:
    #         net = paddle.jit.to_static(net)
    #     # net.train()
    #
    #     # 构建optimizer用于训练
    #     # opt = self.optimizer_info.get_opt(net=net)
    #
    #     for epoch in range(self.step):
    #         if isinstance(self.input_data, paddle.io.DataLoader):  # data_module_type == 'DataLoader'
    #             for i, data_dict in enumerate(self.input_data()):
    #                 logit = net(**data_dict)
    #                 # 构建loss用于训练
    #                 # logit = self.loss_info.get_loss(logit)
    #                 logit = self.loss(logit)
    #                 logit.backward()
    #                 self.optimizer.step()
    #                 self.optimizer.clear_grad()
    #         else:  # data_module_type == 'Dataset'
    #             data_dict = self.input_data[epoch]
    #             logit = net(**data_dict)
    #             # 构建loss用于训练
    #             # logit = self.loss_info.get_loss(logit)
    #             logit = self.loss(logit)
    #             logit.backward()
    #             self.optimizer.step()
    #             self.optimizer.clear_grad()
    #     return logit
