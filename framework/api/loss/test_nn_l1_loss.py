#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.L1Loss
"""

import paddle
from runner import Runner
from base_dygraph_model import Dygraph
import reader
import numpy as np


def test_nn_l1_loss_base():
    """
    test nn.L1Loss base test
    """
    model = Dygraph()
    datareader = np.random.random(size=[2, 10])
    label = np.array([[1.7, 1.0], [0.4, 0.5]]).astype(np.float64)
    loss = paddle.nn.L1Loss
    runner = Runner(datareader, label, model, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict("params_group1", reduction="mean", name=None)
    runner.run()
    expect = [
        2.118743347846277,
        2.1164528078209917,
        2.1141622677957073,
        2.111871727770422,
        2.1095811877451367,
        2.1072906477198514,
        2.1050001076945666,
        2.1027095676692813,
        2.1004190276439965,
        2.0981284876187107,
    ]
    runner.check(expect)
