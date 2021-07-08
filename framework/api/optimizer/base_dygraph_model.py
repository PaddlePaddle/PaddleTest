#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
dygraph model
"""

import paddle
import numpy as np


class Dygraph(paddle.nn.Layer):
    """model"""

    def __init__(self):
        """__init__"""
        paddle.set_default_dtype(np.float64)
        super(Dygraph, self).__init__()
        self.fc = paddle.nn.Linear(in_features=10, out_features=2)

    def forward(self, inputs):
        """forward"""
        output = self.fc(inputs)
        return output
