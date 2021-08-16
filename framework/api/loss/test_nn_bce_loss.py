#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.BCELoss
"""

import paddle
from runner import Runner
from base_dygraph_model import Dygraph
import reader
import numpy as np


def test_nn_bce_loss_base():
    """
    test nn.BCELoss base test
    """
    model = Dygraph()
    datareader = np.random.random(size=[5, 10])
    label = np.random.randint(0, 4, size=[5, 2]).astype(np.float64)
    loss = paddle.nn.BCELoss
    runner = Runner(datareader, label, model, loss)
    runner.softmax = True
    runner.add_kwargs_to_dict("params_group1", weight=None, reduction="mean", name=None)
    runner.run()
    expect = [
        0.6931471805599452,
        0.692812321317948,
        0.6924779430627329,
        0.6921440440621012,
        0.6918106225903953,
        0.6914776769284808,
        0.6911452053637225,
        0.6908132061899631,
        0.6904816777075028,
        0.6901506182230779,
    ]
    runner.check(expect)
