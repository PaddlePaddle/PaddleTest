#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
train builder
"""
import paddle


class BuildTrain(object):
    """BuildTrain"""

    def __init__(self, train_info):
        """init"""
        self.train_info = train_info

    def get_train_step(self):
        """get_train_step"""
        return self.train_info["step"]
