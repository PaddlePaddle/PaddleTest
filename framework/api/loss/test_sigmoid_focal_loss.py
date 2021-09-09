#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.functional.sigmoid_focal_loss
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import numpy as np


@pytest.mark.loss_sigmoid_focal_loss_vartype
def test_nn_sigmoid_focal_loss_base():
    """
    test nn.functional.sigmoid_focal_loss base test
    """
    datareader = np.random.random(size=[5, 10]).astype(np.float32)
    label = np.random.randint(0, 1, size=[5, 2]).astype(np.float32)
    norm = np.array([3.0]).astype(np.float32)
    loss = paddle.nn.functional.sigmoid_focal_loss
    runner = Runner(datareader, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict(
        "params_group1", label=label, normalizer=norm, alpha=0.25, gamma=2.0, reduction="sum", name=None
    )
    runner.add_kwargs_to_dict("params_group3", dtype=np.float32)
    expect = [5.9079647, 5.894992, 5.882022, 5.8690534, 5.8560877, 5.843124, 5.830162, 5.8172035, 5.8042474, 5.791294]
    runner.run(model=Dygraph, expect=expect)
