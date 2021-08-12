#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.MSELoss
"""

import paddle
from runner import Runner
from base_dygraph_model import Dygraph
import reader
import numpy as np


def test_nn_mse_loss_base():
    """
    test nn.MSELoss base test
    """
    model = Dygraph()
    datareader = np.random.random(size=[5, 10])
    label = np.random.randint(0, 1, size=[5, 2]).astype(np.float64)
    loss = paddle.nn.MSELoss
    runner = Runner(datareader, label, model, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict("params_group1", reduction="mean")
    runner.run()
    expect = [
        8.797613809314154,
        8.727873155895239,
        8.65868640960391,
        8.590049170783894,
        8.521957074725274,
        8.454405791386904,
        8.38739102512103,
        8.320908514400106,
        8.254954031545779,
        8.189523382460031,
    ]
    runner.check(expect)
