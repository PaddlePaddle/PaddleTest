#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.functional.binary_cross_entropy
"""

import paddle
from runner import Runner
from base_dygraph_model import Dygraph
import reader
import numpy as np


def test_binary_cross_entropy_base():
    """
    test binary_cross_entropy base test
    """
    model = Dygraph()
    datareader = reader.reader
    label = np.array([[[0, 1]]]).astype(np.float64)
    loss = paddle.nn.functional.binary_cross_entropy
    runner = Runner(datareader, label, model, loss)
    runner.softmax = True
    runner.add_kwargs_to_dict("params_group1", weight=None, reduction="mean", name=None)
    runner.run()
    expect = [
        0.6931471805599453,
        0.6911983087323113,
        0.6892570330858404,
        0.687323323940925,
        0.6853971516773683,
        0.683478486735148,
        0.6815672996151696,
        0.6796635608800039,
        0.6777672411546121,
        0.6758783111270596,
    ]
    runner.check(expect)
