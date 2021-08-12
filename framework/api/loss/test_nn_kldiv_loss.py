#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.KLDivLoss
"""

import paddle
from runner import Runner
from base_dygraph_model import Dygraph
import reader
import numpy as np


def test_nn_kldiv_loss_base():
    """
    test nn.KLDivLoss base test
    """
    model = Dygraph()
    datareader = np.random.random(size=[5, 10]).astype("float64")
    label = np.random.random(size=[5, 2]).astype("float64")
    loss = paddle.nn.KLDivLoss
    runner = Runner(datareader, label, model, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict("params_group1", reduction="mean")
    runner.run()
    expect = [
        -1.4750609889053445,
        -1.4753909973250376,
        -1.4757210057447305,
        -1.476051014164423,
        -1.4763810225841156,
        -1.4767110310038087,
        -1.4770410394235016,
        -1.4773710478431945,
        -1.4777010562628872,
        -1.4780310646825803,
    ]
    runner.check(expect)
