#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.BCELoss
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import numpy as np


@pytest.mark.loss_nn_BCELoss_vartype
def test_nn_bce_loss_base():
    """
    test nn.BCELoss base test
    """
    datareader = np.random.random(size=[5, 10])
    label = np.random.randint(0, 4, size=[5, 2]).astype(np.float64)
    loss = paddle.nn.BCELoss
    runner = Runner(datareader, loss)
    runner.softmax = True
    runner.add_kwargs_to_dict("params_group1", weight=None, reduction="mean", name=None)
    runner.add_kwargs_to_dict("params_group2", label=label)
    expect = [
        0.6931471805599452,
        0.6930619132122062,
        0.6929768953645418,
        0.6928921262003893,
        0.6928076049058101,
        0.6927233306694804,
        0.6926393026826853,
        0.6925555201393077,
        0.6924719822358253,
        0.6923886881712995,
    ]
    runner.run(model=Dygraph, expect=expect)
