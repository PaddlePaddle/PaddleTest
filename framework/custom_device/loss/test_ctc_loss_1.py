#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.functional.ctc_loss
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import numpy as np


@pytest.mark.loss_ctc_loss_parameters
def test_nn_cross_entropy_loss_1():
    """
    reduction='none'
    """
    datareader = np.random.random(size=[5, 2, 10]) * 2 - 1
    label = np.array([[1, 2, 2], [1, 2, 2]]).astype("int32")
    loss = paddle.nn.functional.ctc_loss
    runner = Runner(datareader, loss)
    runner.softmax = False
    runner.static = False
    runner.add_kwargs_to_dict(
        "params_group1",
        labels=label,
        input_lengths=np.array([5, 5]).astype("int64"),
        label_lengths=np.array([3, 3]).astype("int64"),
        blank=0,
        reduction="none",
    )
    runner.add_kwargs_to_dict("params_group3", out_features=3)
    expect = [
        3.5471512942852357,
        3.54066629143771,
        3.5342045336133383,
        3.5277658997178754,
        3.5213502695414136,
        3.5149575237494712,
        3.508587543874176,
        3.5022402123055336,
        3.4959154122827805,
        3.48961302788582,
    ]
    runner.run(model=Dygraph, expect=expect)
