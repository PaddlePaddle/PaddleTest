#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.SmoothL1Loss
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import numpy as np


@pytest.mark.loss_nn_SmoothL1Loss_vartype
def test_nn_smooth_l1_loss_base():
    """
    test nn.SmoothL1Loss base test
    """
    datareader = np.random.random(size=[5, 10]).astype(np.float64)
    label = np.random.rand(5, 2).astype(np.float64)
    loss = paddle.nn.SmoothL1Loss
    runner = Runner(datareader, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict("params_group1", reduction="mean", delta=1.0, name=None)
    runner.add_kwargs_to_dict("params_group2", label=label)
    expect = [
        1.75875173718168,
        1.7571122921724502,
        1.7554728471632206,
        1.7538334021539903,
        1.7521939571447611,
        1.750554512135531,
        1.7489150671263012,
        1.7472756221170715,
        1.745636177107842,
        1.7439967320986116,
    ]
    runner.run(model=Dygraph, expect=expect)
