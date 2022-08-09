#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
train builder
"""
import paddle


class BuildTrain(object):
    """BuildTrain"""

    def __init__(self, step, optimizer):
        """init"""
        self.step = step
        self.optimizer = optimizer
        self.opt_api = self.optimizer.get("opt_api", "paddle.optimizer.SGD")
        self.learning_rate = self.optimizer.get("learning_rate", 0.0001)

    def get_train_optimizer(self):
        """get optimizer"""
        return self.opt_api

    def get_train_step(self):
        """get_train_step"""
        return self.step

    def get_train_lr(self):
        """get_train_lr"""
        return self.learning_rate
