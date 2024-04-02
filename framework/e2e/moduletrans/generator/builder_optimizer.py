#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
loss builder
"""

import numpy as np
import paddle
import diy


class BuildOptimizer(object):
    """BuildData"""

    def __init__(self, optimizer_name, optimizer_param):
        """init"""
        self.optimizer_name = optimizer_name
        self.optimizer_param = optimizer_param

    def get_opt_name(self):
        """get optimizer name"""
        return self.optimizer_name

    def get_opt_param_dict(self):
        """get optimizer param"""
        return self.optimizer_param

    def get_opt(self, net):
        """get optimizer"""
        opt = eval(self.optimizer_name)(net, **self.optimizer_param)
        return opt
