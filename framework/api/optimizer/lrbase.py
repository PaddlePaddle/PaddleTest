#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
case runner
"""
import copy
import logging
import paddle
import numpy as np

seed = 33
np.random.seed(seed)
paddle.seed(seed)


class Runner(object):
    """Runner"""

    def __init__(self, paddle_lr, naive_func):
        """init"""
        # self.reader = paddle.to_tensor(reader)
        # self.model = model
        # self.optimizer = optimizer
        self.paddle_lr = paddle_lr
        self.naive_func = naive_func
        self.debug = False
        self.lr_0 = 0.01
        self.run_time = 20
        self.delta = 1e-10
        self.rtol = 1e-11
        self.result = []
        # 传参工具
        self.kwargs_dict = {"params_group1": {}, "params_group2": {}, "params_group3": {}}

    def add_kwargs_to_dict(self, group_name, **kwargs):
        """
        params dict tool
        """
        self.kwargs_dict[group_name] = kwargs

    def run(self):
        """run your models"""
        # res_lr = self.paddle_lr(self.lr_0, **self.kwargs_dict["params_group1"])
        self.paddle_func = copy.deepcopy(self.paddle_lr)
        # res_lr = self.lr_0
        res_lr = self.paddle_func(learning_rate=self.lr_0, **self.kwargs_dict["params_group1"])
        res_lr.get_lr()
        exp_lr = self.lr_0
        epoch = 0
        res_list = []
        exp_list = []
        # logging.info('exp_lr is: {}'.format(exp_lr))
        # logging.info('res_lr is: {}'.format(res_lr.get_lr()))
        for i in range(0, self.run_time):
            epoch += 1
            # res_lr = self.paddle_func(res_lr, **self.kwargs_dict["params_group1"])
            exp_lr = self.naive_func(lr_last=exp_lr, lr_0=self.lr_0, epoch=epoch, **self.kwargs_dict["params_group1"])
            res_lr.step()
            # epoch += 1
            logging.info("exp_lr is: {}".format(exp_lr))
            logging.info("res_lr is: {}".format(res_lr.get_lr()))
            res_list.append(res_lr.get_lr())
            exp_list.append(exp_lr)
            # assert exp_lr == res_lr.get_lr()
        self.check(result=res_list, expect=exp_list, delta=self.delta, rtol=self.rtol)

        # out = self.model(self.reader)
        # loss = paddle.mean(out)
        # loss.backward()
        # self.optimizer.step()
        # self.optimizer.clear_grad()
        # if self.debug:
        #     print(loss)
        # self.result.append(loss.numpy()[0])

    def check(self, result=None, expect=None, delta=1e-6, rtol=1e-7):
        """
        check result
        """
        if result is None:
            raise Exception("Model result is None， check your code")
        if self.debug:
            print(result)
        try:
            assert np.allclose(result, expect, atol=delta, rtol=rtol), "Error in check loss"
        except Exception as e:
            print(e)
            print("expect loss is {}".format(expect))
            print("Model result is {}".format(result))
            assert False
