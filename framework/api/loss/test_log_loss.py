#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.functional.log_loss
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import numpy as np


@pytest.mark.loss_log_loss_vartype
def test_log_loss_base():
    """
    test log_loss base test
    """
    datareader = np.random.random(size=[10, 10]).astype(np.float32)
    label = np.random.random(size=[10, 1]).astype(np.float32)
    loss = paddle.nn.functional.log_loss
    runner = Runner(datareader, loss)
    runner.static = False
    runner.softmax = True
    runner.add_kwargs_to_dict("params_group1", label=label, epsilon=0.0001, name=None)
    expect = [5.630299, 5.630299, 5.630299, 5.630299, 5.630299, 5.630299, 5.630299, 5.630299, 5.630299, 5.630299]
    runner.run(model=Dygraph, expect=expect, dtype=np.float32, in_features=10, out_features=1)
