#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
loss builder
"""

import numpy as np
import paddle
import diy


class BuildLoss(object):
    """BuildData"""

    def __init__(self, loss_name, loss_param):
        """init"""
        self.loss_name = loss_name
        self.loss_param = loss_param

    def get_loss_name(self):
        """get loss name"""
        return self.loss_name

    def get_loss_param_dict(self):
        """get loss name"""
        return self.loss_param

    def get_loss(self, logit):
        """get loss"""
        logit = eval(self.loss_name)(logit, **self.loss_param)
        return logit
