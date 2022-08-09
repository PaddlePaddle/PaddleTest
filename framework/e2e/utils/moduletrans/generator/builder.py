#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
builder
"""
import numpy as np
import paddle
import moduletrans
import generator.builder_layer as builder_layer
import generator.builder_data as builder_data
import generator.builder_train as builder_train
import tool
from logger import logger


class BuildModuleTest(object):
    """BuildModuleTest"""

    def __init__(self, case):
        """init"""
        paddle.seed(33)
        np.random.seed(33)
        self.case = moduletrans.ModuleTrans(case)
        self.logger = self.case.logger
        self.layer_info = builder_layer.BuildLayer(*self.case.get_layer_info())
        # self.layer = self.layer_info.get_layer()
        # self.logger.get_log().info('module层级结构: {}'.format(self.layer))
        # print(self.layer.get_layer())
        # net = self.layer.get_layer()

        # data_type, data_input = self.case.get_data_info()
        # print('data type is: ', data_type)
        # print('data input is: ', data_input)
        self.data_info = builder_data.BuildData(*self.case.get_data_info())
        print(self.data_info.get_single_paddle_data())
        self.input_data = self.data_info.get_single_paddle_data()

        # self.out = net(**self.input_data)
        # print('out is: ', self.out)
        self.train_info = builder_train.BuildTrain(*self.case.get_train_info())

    def train(self, to_static=False):
        """dygraph or static train"""
        paddle.seed(33)
        np.random.seed(33)
        # input_data = self.data_info.get_single_paddle_data()
        net = self.layer_info.get_layer()

        if to_static:
            net = paddle.jit.to_static(net)

        opt = eval(self.train_info.get_train_optimizer())(
            learning_rate=self.train_info.get_train_lr(), parameters=net.parameters()
        )
        # dygraph train
        for epoch in range(self.train_info.get_train_step()):
            logit = net(**self.input_data)
            logit.backward()
            opt.step()
            opt.clear_grad()
        return logit

    def predict(self, to_static=False):
        """predict"""
        paddle.seed(33)
        np.random.seed(33)
        net = self.layer_info.get_layer()
        if to_static:
            net = paddle.jit.to_static(net)
        logit = net(**self.input_data)
        return logit

    def dygraph_train_test(self):
        """dygraph train test"""
        pass

    def dygraph_predict_test(self):
        """dygraph predict test"""
        pass

    def dygraph_to_static_train_test(self):
        """dygraph_to_static train test"""
        dygraph_out = self.train(to_static=False)
        self.logger.get_log().info("dygraph to static train acc-test start~")
        self.logger.get_log().info("dygraph_out is: {}".format(dygraph_out))
        static_out = self.train(to_static=True)
        self.logger.get_log().info("static_out is: {}".format(static_out))
        tool.compare(dygraph_out, static_out, delta=1e-8, rtol=1e-8)
        self.logger.get_log().info("dygraph to static train acc-test Success!!!~~")

    def dygraph_to_static_predict_test(self):
        """dygraph_to_static predict test"""
        dygraph_out = self.predict(to_static=False)
        self.logger.get_log().info("dygraph to static predict acc-test start~")
        self.logger.get_log().info("dygraph_out is: {}".format(dygraph_out))
        static_out = self.predict(to_static=True)
        self.logger.get_log().info("static_out is: {}".format(static_out))
        tool.compare(dygraph_out, static_out, delta=1e-8, rtol=1e-8)
        self.logger.get_log().info("dygraph to static predict acc-test Success!!!~~")
