#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
loss builder
"""

import numpy as np
import paddle
import tool


class BuildLoss(object):
    """BuildData"""

    def __init__(self, loss_list):
        """init"""
        self.loss_list = loss_list

    def get_loss_list(self):
        """get loss"""
        return self.loss_list
