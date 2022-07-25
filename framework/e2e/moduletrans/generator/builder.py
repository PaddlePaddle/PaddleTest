#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
builder
"""
import os
from collections import Iterable
import numpy as np
import paddle
import paddle.inference as paddle_infer
import moduletrans
import generator.builder_layer as builder_layer
import generator.builder_loss as builder_loss
import generator.builder_data as builder_data
import generator.builder_optimizer as builder_optimizer
import generator.builder_train as builder_train
import tool
import diy


class BuildModuleTest(object):
    """BuildModuleTest"""

    def __init__(self, case):
        """init"""
        paddle.seed(33)
        np.random.seed(33)
        self.save_path = os.path.join(os.getcwd(), "save_test")
        self.exp_path = os.path.join(os.getcwd(), "ground_truth")
        self.case = moduletrans.ModuleTrans(case)
        self.case_name = self.case.case_name
        self.logger = self.case.logger

        self.data_info = builder_data.BuildData(self.case.get_data_info())
        self.input_data = self.data_info.get_single_data()

        self.layer_info = builder_layer.BuildLayer(*self.case.get_layer_info())

        self.loss_info = builder_loss.BuildLoss(*self.case.get_loss_info())

        self.optimizer_info = builder_optimizer.BuildOptimizer(*self.case.get_opt_info())

        self.train_info = builder_train.BuildTrain(self.case.get_train_info())

    def train(self, to_static=False):
        """dygraph or static train"""
        paddle.enable_static()
        paddle.disable_static()
        paddle.seed(33)
        np.random.seed(33)
        net = self.layer_info.get_layer()

        if to_static:
            net = paddle.jit.to_static(net)
        # net.train()

        # 构建optimizer用于训练
        opt = self.optimizer_info.get_opt(net=net)

        for epoch in range(self.train_info.get_train_step()):
            if isinstance(self.input_data, paddle.io.DataLoader):  # data_module_type == 'DataLoader'
                for i, data_dict in enumerate(self.input_data()):
                    logit = net(**data_dict)
                    # 构建loss用于训练
                    logit = self.loss_info.get_loss(logit)
                    # self.logger.get_log().info('logit is: {}'.format(logit))
                    logit.backward()
                    opt.step()
                    opt.clear_grad()
            else:  # data_module_type == 'Dataset'
                data_dict = self.input_data[epoch]
                # self.logger.get_log().info('data dict for train is: {}'.format(data_dict))
                # self.logger.get_log().info('data for train is: {}'.format(data_dict['inputs']['image']))
                logit = net(**data_dict)
                # 构建loss用于训练
                logit = self.loss_info.get_loss(logit)
                # self.logger.get_log().info("logit is: {}".format(logit))
                logit.backward()
                opt.step()
                opt.clear_grad()
        return logit

    def predict(self, to_static=False):
        """predict"""
        paddle.enable_static()
        paddle.disable_static()
        paddle.seed(33)
        np.random.seed(33)
        data_dict = self.input_data.__getitem__(0)
        net = self.layer_info.get_layer()
        if to_static:
            net = paddle.jit.to_static(net)
        net.eval()
        logit = net(**data_dict)
        return logit

    def jit_save(self):
        """jit.save(layer)"""
        paddle.enable_static()
        paddle.disable_static()
        paddle.seed(33)
        np.random.seed(33)
        net = self.layer_info.get_layer()
        net = paddle.jit.to_static(net)
        net.eval()
        data_dict = self.input_data.__getitem__(0)
        net(**data_dict)
        paddle.jit.save(net, path=os.path.join(self.save_path, "jit_save", self.case_name))

    def infer_load(self):
        """infer load (layer)"""
        paddle.enable_static()
        paddle.disable_static()
        paddle.seed(33)
        np.random.seed(33)
        config = paddle_infer.Config(
            os.path.join(self.save_path, "jit_save", self.case_name + ".pdmodel"),
            os.path.join(self.save_path, "jit_save", self.case_name + ".pdiparams"),
        )
        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()

        ##
        input_dict = self.input_data.__getitem__(0)
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

    def build_dygraph_train_ground_truth(self, mode="numpy"):
        """dygraph train test"""
        if not os.path.exists(os.path.join(self.exp_path, self.case_name)):
            os.makedirs(os.path.join(self.exp_path, self.case_name))
        self.logger.get_log().info("dygraph train build ground truth start~")
        exp_out = self.train(to_static=False)
        if mode == "numpy":
            np.save(os.path.join(self.exp_path, self.case_name, "dygraph_train_test.npy"), exp_out.numpy())
        else:
            self.logger.get_log().error("unknown save type for build_dygraph_train_ground_truth!!!~")

    def dygraph_train_test(self, delta=1e-8, rtol=1e-8):
        """dygraph train test"""
        self.logger.get_log().info("dygraph train acc-test start~")
        res_out = self.train(to_static=False)
        # self.logger.get_log().info("dygraph_out is: {}".format(res_out))
        exp_out = np.load(os.path.join(self.exp_path, self.case_name, "dygraph_train_test.npy"))
        tool.compare(res_out, exp_out, delta=delta, rtol=rtol)

    def dygraph_predict_test(self, delta=1e-8, rtol=1e-8):
        """dygraph predict test"""
        pass

    def dygraph_to_static_train_test(self, delta=1e-8, rtol=1e-8):
        """dygraph_to_static train test"""
        self.logger.get_log().info("dygraph to static train acc-test start~")
        dygraph_out = self.train(to_static=False)
        self.logger.get_log().info("dygraph_out is: {}".format(dygraph_out))
        static_out = self.train(to_static=True)
        # self.logger.get_log().info("static_out is: {}".format(static_out))
        tool.compare(static_out, dygraph_out, delta=delta, rtol=rtol)
        # self.logger.get_log().info("dygraph to static train acc-test Success~~")

    def dygraph_to_static_predict_test(self, delta=1e-8, rtol=1e-8):
        """dygraph_to_static predict test"""
        self.logger.get_log().info("dygraph to static predict acc-test start~")
        dygraph_out = self.predict(to_static=False)
        # self.logger.get_log().info("dygraph_out is: {}".format(dygraph_out))
        static_out = self.predict(to_static=True)
        # self.logger.get_log().info("static_out is: {}".format(static_out))
        tool.compare(static_out, dygraph_out, delta=delta, rtol=rtol)
        # self.logger.get_log().info("dygraph to static predict acc-test Success~~")

    def dygraph_to_infer_predict_test(self, acc_test=False, delta=1e-5, rtol=1e-5):
        """dygraph_to_static predict test"""
        dygraph_out = self.predict(to_static=False)
        self.logger.get_log().info("dygraph to infer export test start~")
        # self.logger.get_log().info("dygraph_out is: {}".format(dygraph_out))
        self.jit_save()
        self.logger.get_log().info("dygraph jit.save Success~~")
        if acc_test:
            self.logger.get_log().info("dygraph to infer predict acc-test start~")
            infer_out = self.infer_load()
            # self.logger.get_log().info("static_out is: {}".format(infer_out))
            tool.compare(infer_out, dygraph_out, delta=delta, rtol=rtol)
            # self.logger.get_log().info("dygraph to infer predict acc-test Success~~")
