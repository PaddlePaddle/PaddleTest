#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
train 方法
"""
import numpy as np
import paddle
from xtools import reset


class LayerTrain(object):
    """
    构建Layer训练的通用类
    """

    def __init__(self, net, data, optimizer, loss, step):
        """
        初始化
        """
        self.net = net
        self.data = data
        self.optimizer = optimizer
        self.loss = loss
        self.step = step
        self.seed = 33

    def reset(self):
        """
        重置模型图
        :return:
        """
        paddle.enable_static()
        paddle.disable_static()
        paddle.seed(self.seed)
        np.random.seed(self.seed)

    def dy_train(self):
        """dygraph train"""
        reset(self.seed)

        # net = self.layer_info.get_layer()
        net = self.net.get_layer()
        net.train()
        # 构建optimizer用于训练
        opt = self.optimizer.get_opt(net=net)

        for epoch in range(self.step):
            # data_module_type == 'Dataset'
            data_dict = self.data[epoch]
            logit = net(**data_dict)
            # 构建loss用于训练
            # logit = self.loss_info.get_loss(logit)
            logit = self.loss.get_loss(logit)
            logit.backward()
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
                logit = self.loss.get_loss(logit)
                logit.backward()
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
            logit = self.loss.get_loss(logit)
            logit.backward()
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
                logit = self.loss.get_loss(logit)
                logit.backward()
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
