#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
Layer trans
"""

import os
from logger import logger


class ModuleTrans(object):
    """Module Trans"""

    def __init__(self, case):
        """init module"""
        self.case = case["info"]
        self.case_name = case["name"]
        # self.case = self.case["paddle"]
        self.logger = logger

        self.logger.get_log().info(self.case.get("desc", "没有描述"))

        self.layer = self.case["Layer"]
        self.data_loader = self.case["DataGenerator"]
        # self.data = self.case["Data"]
        self.loss = self.case["Loss"]
        self.optimizer = self.case["optimizer"]
        self.train = self.case["Train"]

    def get_layer_info(self):
        """get layer info"""
        repo = self.layer["repo"]
        if repo not in ["DIY", "PaddleOCR", "PaddleSeg", "PaddleClas", "PaddleDetection"]:
            self.logger.get_log().warning("{} 未知模型repo".format(repo))
        layer_name = self.layer["layer_name"]
        layer_param = self.layer["params"]
        return repo, layer_name, layer_param

    def get_data_info(self):
        """get data info"""
        return self.data_loader

    def get_loss_info(self):
        """get loss info"""
        loss_name = self.loss["loss_name"]
        loss_param = self.loss["params"]
        return loss_name, loss_param

    def get_opt_info(self):
        """get optimizer info"""
        optimizer_name = self.optimizer["optimizer_name"]
        optimizer_param = self.optimizer["params"]
        return optimizer_name, optimizer_param

    def get_train_info(self):
        """get train info"""
        return self.train

    def get_ground_truth_path(self):
        """get path"""
        layer_name = self.get_layer_info()[1]
        str_list = layer_name.split(".")
        save_test_path = os.path.join(os.getcwd(), "save_test")
        ground_truth_path = os.path.join(os.getcwd(), "ground_truth")
        for i in str_list:
            save_test_path = os.path.join(save_test_path, i)
            ground_truth_path = os.path.join(ground_truth_path, i)
        return save_test_path, ground_truth_path
