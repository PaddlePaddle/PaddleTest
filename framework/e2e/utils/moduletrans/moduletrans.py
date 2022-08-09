#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
Layer trans
"""

import random
from inspect import isclass
import paddle
import numpy as np
from logger import logger


class ModuleTrans(object):
    """Module Trans"""

    def __init__(self, case):
        """init module"""
        self.case = case["info"]
        self.case_name = case["name"]
        self.paddle = self.case["paddle"]
        self.logger = logger

        self.logger.get_log().info(self.case.get("desc", "没有描述"))

        self.layer = self.paddle["Layer"]
        self.data_loader = self.paddle["DataLoader"]
        self.data = self.paddle["Data"]
        self.loss = self.paddle["Loss"]
        self.train = self.paddle["Train"]

    def get_layer_info(self):
        """get layer info"""
        repo = self.layer["repo"]
        if repo not in ["DIY", "PaddleOCR", "PaddleSeg"]:
            self.logger.get_log().warning("{} 未知模型repo".format(repo))
        layer_name = self.layer["layer_name"]
        layer_param = self.layer["params"]
        return repo, layer_name, layer_param

    def get_data_info(self):
        """get data info"""
        if self.data_loader not in ["single", "DataLoader"]:
            self.logger.get_log().warning("{} 未知模型repo".format(self.data_loader))
        return self.data_loader, self.data

    def get_loss_info(self):
        """get loss info"""
        loss_api = self.loss["loss_api"]
        return loss_api

    def get_train_info(self):
        """get train info"""
        train_step = self.train["step"]
        optimizer_info = self.train["optimizer"]
        return train_step, optimizer_info
