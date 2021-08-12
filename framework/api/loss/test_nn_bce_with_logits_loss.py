#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.BCEWithLogitsLoss
"""

import paddle
from runner import Runner
from base_dygraph_model import Dygraph
import reader
import numpy as np


def test_nn_bce_with_logits_loss_base():
    """
    test nn.BCEWithLogitsLoss base test
    """
    model = Dygraph()
    datareader = np.random.random(size=[5, 10])
    label = np.random.randint(0, 4, size=[5, 2]).astype(np.float64)
    loss = paddle.nn.BCEWithLogitsLoss
    runner = Runner(datareader, label, model, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict("params_group1", weight=None, reduction="mean", name=None)
    runner.run()
    expect = [
        -1.063874428870802,
        -1.0643588797694155,
        -1.0648432592922026,
        -1.0653275675130562,
        -1.0658118045057754,
        -1.066295970344068,
        -1.06678006510155,
        -1.0672640888517468,
        -1.0677480416680907,
        -1.068231923623924,
    ]
    runner.check(expect)
