#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.CrossEntropyLoss
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import numpy as np


@pytest.mark.loss_nn_CrossEntropyLoss_vartype
def test_nn_cross_entropy_loss_base():
    """
    test nn.CrossEntropyLoss base test
    """
    datareader = np.random.random(size=[5, 10])
    label = np.random.randint(0, 1, size=(5)).astype(np.int64)
    loss = paddle.nn.CrossEntropyLoss
    runner = Runner(datareader, loss)
    runner.case_name = "_base"
    runner.softmax = False
    runner.add_kwargs_to_dict(
        "params_group1", weight=None, ignore_index=-100, reduction="mean", soft_label=False, axis=-1, name=None
    )
    runner.add_kwargs_to_dict("params_group2", label=label)
    runner.add_kwargs_to_dict("params_group3", dtype=np.float64, in_features=10, out_features=2)
    expect = [
        0.6931471805599453,
        0.6915090863883171,
        0.6898763866549043,
        0.6882490635524297,
        0.6866270993032069,
        0.6850104761594786,
        0.6833991764037487,
        0.6817931823491103,
        0.6801924763395683,
        0.6785970407503574,
    ]
    runner.run(model=Dygraph, expect=expect)
