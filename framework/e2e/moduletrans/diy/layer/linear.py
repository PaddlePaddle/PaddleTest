#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
linear
"""
import paddle


class Linear(paddle.nn.Layer):
    """Linear"""

    def __init__(self, in_features=3, out_features=10):
        """init"""
        super(Linear, self).__init__()
        self.fc0 = paddle.nn.Linear(in_features=in_features, out_features=out_features)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, x):
        """forward"""
        out = self.fc0(x)
        out = self.sigmoid(out)
        return out
