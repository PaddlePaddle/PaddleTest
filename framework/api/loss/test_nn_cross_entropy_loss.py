#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.CrossEntropyLoss
"""

import paddle
from runner import Runner
from base_dygraph_model import Dygraph
import reader
import numpy as np


def test_nn_cross_entropy_loss_base():
    """
    test nn.CrossEntropyLoss base test
    """
    model = Dygraph()
    datareader = np.random.random(size=[5, 10])
    label = np.random.randint(0, 1, size=(5)).astype(np.int64)
    loss = paddle.nn.CrossEntropyLoss
    runner = Runner(datareader, label, model, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict(
        "params_group1", weight=None, ignore_index=-100, reduction="mean", soft_label=False, axis=-1, name=None
    )
    runner.run()
    expect = [
        0.6931471805599453,
        0.6912159391987173,
        0.6892923664054587,
        0.6873764316147266,
        0.6854681043190555,
        0.6835673540698795,
        0.6816741504784372,
        0.679788463216662,
        0.6779102620180553,
        0.6760395166785476,
    ]
    runner.check(expect)
