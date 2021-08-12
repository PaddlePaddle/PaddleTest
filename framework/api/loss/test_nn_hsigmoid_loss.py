#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.HSigmoidLoss
"""

import paddle
from runner import Runner
from base_dygraph_model import Dygraph
import reader
import numpy as np


def test_nn_hsigmoid_loss_base():
    """
    test nn.HSigmoidLoss base test
    """
    model = Dygraph()
    datareader = np.random.random(size=[3, 1, 10])
    label = np.random.randint(0, 5, size=(4)).astype(np.int64)
    loss = paddle.nn.HSigmoidLoss
    runner = Runner(datareader, label, model, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict(
        "params_group1",
        feature_size=3,
        num_classes=5,
        weight_attr=None,
        bias_attr=None,
        is_custom=False,
        is_sparse=False,
        name=None,
    )
    runner.run()
    expect = [
        np.array([4.71086874]),
        np.array([2.41106201]),
        np.array([2.8604591]),
        np.array([1.87135578]),
        np.array([1.57817298]),
        np.array([4.26013635]),
        np.array([1.16507945]),
        np.array([3.51465264]),
        np.array([4.06945707]),
        np.array([2.76762656]),
    ]

    runner.check(expect)
