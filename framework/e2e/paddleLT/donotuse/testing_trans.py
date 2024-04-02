#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
测试项目配置相关
"""

import os
import platform
import time
import pytest
import allure
from tools.yaml_loader import YamlLoader


class TrainTrans(object):
    """获取单个测试项目中的测试参数信息"""

    def __init__(self, testing):
        """init"""
        self.testing = testing

    def get_loss_info(self, logit):
        """get loss info"""
        # loss_name = self.testing.get("train").get("Loss").get("loss_name")
        # loss_param = self.testing.get("train").get("Loss").get("params")

        loss_name = self.testing.get("Loss").get("loss_name")
        loss_param = self.testing.get("Loss").get("params")
        logit = eval(loss_name)(logit, **loss_param)
        return logit

    def get_opt_info(self, net):
        """get optimizer info"""
        # optimizer_name = self.testing.get("train").get("optimizer").get("optimizer_name")
        # optimizer_param = self.testing.get("train").get("optimizer").get("params")

        optimizer_name = self.testing.get("optimizer").get("optimizer_name")
        optimizer_param = self.testing.get("optimizer").get("params")

        opt = eval(optimizer_name)(net, **optimizer_param)
        return opt

    def get_train_step(self):
        """get_train_step"""
        # return self.testing.get("train").get("step")
        return self.testing.get("step")

    def get_model_dtype(self):
        """get_train_step"""
        return self.testing.get("model_dtype")

    # def get_testing_chain(self):
    #     """get testing chain"""
    #     testing_list = self.testing.get("testing_chain")
    #     return testing_list

    # def get_train_info(self):
    #     """get train info"""
    #     train_info = self.testing.get("train")
    #     return train_info

    # def get_threshold(self):
    #     """get train info"""
    #     delta = self.testing.get("threshold").get("delta")
    #     rtol = self.testing.get("threshold").get("rtol")
    #     return delta, rtol
