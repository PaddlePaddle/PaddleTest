#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.NLLLoss
"""

import paddle
from runner import Runner
from base_dygraph_model import Dygraph
import reader
import numpy as np


def test_nn_nll_loss_base():
    """
    test nll loss base test
    """
    model = Dygraph()
    datareader = np.random.random(size=[5, 10])
    label = np.random.randint(0, 1, size=[5, 5]).astype(np.int64)
    loss = paddle.nn.NLLLoss
    runner = Runner(datareader, label, model, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict("params_group1", weight=None, ignore_index=-100, reduction="mean", name=None)
    runner.run()
    expect = [
        -2.9010277162615785,
        -2.9048940409176547,
        -2.9087603655737295,
        -2.9126266902298057,
        -2.916493014885881,
        -2.9203593395419563,
        -2.924225664198032,
        -2.9280919888541073,
        -2.931958313510183,
        -2.9358246381662587,
    ]
    runner.check(expect)
