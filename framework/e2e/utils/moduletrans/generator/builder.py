#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
builder
"""
import os
import numpy as np
import paddle
import paddle.inference as paddle_infer
import moduletrans
import generator.builder_layer as builder_layer
import generator.builder_loss as builder_loss
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
        self.save_path = os.path.join(os.getcwd(), "save_test")
        self.case = moduletrans.ModuleTrans(case)
        self.case_name = self.case.case_name
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

        self.loss_info = builder_loss.BuildLoss(self.case.get_loss_info())
        self.loss_list = self.loss_info.get_loss_list()

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
            # 按照list顺序构建组合loss
            for l in self.loss_list:
                print("l is: ", l)
                logit = eval(l)
            # logit = eval('logit[0] + logit[3]')
            # logit.backward()
            # loss_list = ['logit[0] + logit[3]', 'paddle.nn.functional.softmax(logit)']
            # for i in loss_list:
            #     logit = eval(i)
            # logit = logit[1] + logit[4]
            # print('logit shape is: ', logit.shape)
            # logit = paddle.nn.functional.softmax(logit)
            # print('logit softmax shape is: ', logit.shape)
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
        net.eval()
        logit = net(**self.input_data)
        return logit

    def jit_save(self):
        """jit.save(layer)"""
        paddle.seed(33)
        np.random.seed(33)
        net = self.layer_info.get_layer()
        net = paddle.jit.to_static(net)
        net.eval()
        net(**self.input_data)
        # inputspec_list = self.data_info.get_single_inputspec()
        # paddle.jit.save(net, path=os.path.join(self.save_path, 'jit_save', self.case_name), input_spec=inputspec_list)
        paddle.jit.save(net, path=os.path.join(self.save_path, "jit_save", self.case_name))

    def infer_load(self):
        """infer load (layer)"""
        paddle.seed(33)
        np.random.seed(33)
        config = paddle_infer.Config(
            os.path.join(self.save_path, "jit_save", self.case_name + ".pdmodel"),
            os.path.join(self.save_path, "jit_save", self.case_name + ".pdiparams"),
        )
        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()

        ##
        input_dict = self.input_data
        inputs_key = sorted(input_dict.keys())

        input_list = []
        for k in inputs_key:
            input_list.append(input_dict[k])

        for i, name in enumerate(input_names):
            input_handle = predictor.get_input_handle(name)
            if isinstance(input_list[i], int):
                input_tmp = np.array(input_list[i])
            elif isinstance(input_list[i], paddle.Tensor):
                input_tmp = input_list[i].numpy()

            input_handle.copy_from_cpu(input_tmp)

        predictor.run()
        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        infer_res = output_handle.copy_to_cpu()
        return infer_res

    def dygraph_train_test(self, delta=1e-8, rtol=1e-8):
        """dygraph train test"""
        pass

    def dygraph_predict_test(self, delta=1e-8, rtol=1e-8):
        """dygraph predict test"""
        pass

    def dygraph_to_static_train_test(self, delta=1e-8, rtol=1e-8):
        """dygraph_to_static train test"""
        dygraph_out = self.train(to_static=False)
        self.logger.get_log().info("dygraph to static train acc-test start~")
        self.logger.get_log().info("dygraph_out is: {}".format(dygraph_out))
        static_out = self.train(to_static=True)
        self.logger.get_log().info("static_out is: {}".format(static_out))
        tool.compare(dygraph_out, static_out, delta=delta, rtol=rtol)
        self.logger.get_log().info("dygraph to static train acc-test Success!!!~~")

    def dygraph_to_static_predict_test(self, delta=1e-8, rtol=1e-8):
        """dygraph_to_static predict test"""
        dygraph_out = self.predict(to_static=False)
        self.logger.get_log().info("dygraph to static predict acc-test start~")
        self.logger.get_log().info("dygraph_out is: {}".format(dygraph_out))
        static_out = self.predict(to_static=True)
        self.logger.get_log().info("static_out is: {}".format(static_out))
        tool.compare(dygraph_out, static_out, delta=delta, rtol=rtol)
        self.logger.get_log().info("dygraph to static predict acc-test Success!!!~~")

    def dygraph_to_infer_predict_test(self, delta=1e-5, rtol=1e-5):
        """dygraph_to_static predict test"""
        dygraph_out = self.predict(to_static=False)
        self.logger.get_log().info("dygraph to infer predict acc-test start~")
        self.logger.get_log().info("dygraph_out is: {}".format(dygraph_out))
        self.jit_save()
        self.logger.get_log().info("dygraph jit.save Success!!!~~")
        infer_out = self.infer_load()
        self.logger.get_log().info("static_out is: {}".format(infer_out))
        tool.compare(dygraph_out, infer_out, delta=delta, rtol=rtol)
        self.logger.get_log().info("dygraph to infer predict acc-test Success!!!~~")
