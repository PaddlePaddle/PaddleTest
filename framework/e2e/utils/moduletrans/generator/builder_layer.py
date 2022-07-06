#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
layer_builder
"""

import paddle
import diy
import ppdet


class BuildLayer(object):
    """BuildLayer"""

    def __init__(self, repo, layer_name, layer_param):
        """init"""
        self.repo = repo
        self.layer_name = layer_name
        self.layer_param = layer_param

    def get_layer(self):
        """get_layer"""
        if self.repo == "DIY":
            layer = eval(self.layer_name)(**self.layer_param)
        else:
            # layer = eval(self.repo + "." + self.layer_name)(**self.layer_param)
            layer = eval(self.layer_name)(**self.layer_param)
        return layer
