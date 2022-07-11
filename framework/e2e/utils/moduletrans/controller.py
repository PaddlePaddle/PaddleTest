#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
controller
"""

import random
from inspect import isclass
import paddle
import numpy as np
from logger import logger
import generator


class ControlTrans(object):
    """Control Trans"""

    def __init__(self, case, controller):
        """init module"""
        self.control = controller["info"]
        self.case = case
        self.case_name = case["name"]
        self.logger = logger

        self.logger.get_log().info(self.control.get("desc", "没有描述"))
        print("self.control is: ", self.control)

        self.test = self.control["test"]
        self.module = generator.builder.BuildModuleTest(self.case)
        # self.data_loader = self.paddle["DataLoader"]
        # self.data = self.paddle["Data"]
        # self.loss = self.paddle["Loss"]
        # self.train = self.paddle["Train"]
        self.test_map = {
            "dygraph_train_test": self.module.dygraph_train_test,
            "dygraph_to_static_train_test": self.module.dygraph_to_static_train_test,
            "dygraph_predict_test": self.module.dygraph_predict_test,
            "dygraph_to_static_predict_test": self.module.dygraph_to_static_predict_test,
            "dygraph_to_infer_predict_test": self.module.dygraph_to_infer_predict_test,
        }

    def run_test(self):
        """run some test"""
        for k, v in self.test.items():
            self.logger.get_log().info("k is: ", k)
            self.logger.get_log().info("v is: ", v)
            self.test_map[k](v["delta"], v["rtol"])
