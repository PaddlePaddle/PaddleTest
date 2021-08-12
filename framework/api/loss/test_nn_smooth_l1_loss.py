#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.SmoothL1Loss
"""

import paddle
from runner import Runner
from base_dygraph_model import Dygraph
import reader
import numpy as np


def test_nn_smooth_l1_loss_base():
    """
    test nn.SmoothL1Loss base test
    """
    model = Dygraph()
    datareader = np.random.random(size=[5, 10]).astype(np.float64)
    label = np.random.rand(5, 2).astype(np.float64)
    loss = paddle.nn.SmoothL1Loss
    runner = Runner(datareader, label, model, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict("params_group1", reduction="mean", delta=1.0, name=None)
    runner.run()
    expect = [
        2.0292157451201143,
        2.0272825827920764,
        2.025349420464039,
        2.023416258136001,
        2.0214830958079633,
        2.0195499334799254,
        2.017616771151888,
        2.01568360882385,
        2.013750446495812,
        2.0118172841677744,
    ]
    runner.check(expect)
