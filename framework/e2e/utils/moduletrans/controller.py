#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
controller
"""

import random
from inspect import isclass
import traceback
import paddle
import numpy as np
from logger import logger
import generator


class ControlModuleTrans(object):
    """Control Trans"""

    def __init__(self, case):
        """init module"""
        # self.control = case["info"]["test"]
        self.case = case
        self.case_name = case["name"]
        self.logger = logger

        # self.logger.get_log().info(self.case.get("desc", "没有描述"))
        self.test = case["info"]["test"]
        self.module = generator.builder.BuildModuleTest(self.case)

        self.test_map = {
            "dygraph_train_test": self.module.dygraph_train_test,
            "dygraph_to_static_train_test": self.module.dygraph_to_static_train_test,
            "dygraph_predict_test": self.module.dygraph_predict_test,
            "dygraph_to_static_predict_test": self.module.dygraph_to_static_predict_test,
            "dygraph_to_infer_predict_test": self.module.dygraph_to_infer_predict_test,
            "build_dygraph_train_ground_truth": self.module.build_dygraph_train_ground_truth,
        }

    def run_test(self):
        """run some test"""
        exc = 0
        fail_test_list = []
        for k, v in self.test.items():
            # self.logger.get_log().info("k is: {}".format(k))
            # self.logger.get_log().info("v is: {}".format(v))
            try:
                self.test_map[k](**self.test[k])
            except Exception:
                self.logger.get_log().info("{} Failed!!!~~".format(k))
                bug_trace = traceback.format_exc()
                logger.get_log().warn(bug_trace)
                exc += 1
                fail_test_list.append(k)
            else:
                self.logger.get_log().info("{} Success~~".format(k))
            # self.test_map[k](v["delta"], v["rtol"])
            # self.logger.get_log().info(bug_trace)

        if exc > 0:
            # raise Exception(bug_trace)
            raise Exception("failed test is: {}".format(fail_test_list))
