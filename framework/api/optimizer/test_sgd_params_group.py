#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test sgd support params groups case
"""
import paddle
import pytest
from runner import Runner
from linear_dygraph_model import LinearNet
import reader


@pytest.mark.api_optimizer_sgd_params_group_vartype
def test_sgd_params_group_base():
    """
    test sgd base test
    """
    model = LinearNet()
    datareader = reader.reader
    optimizer = paddle.optimizer.SGD(
        learning_rate=0.01,
        parameters=[
            {"params": model.fc1.parameters()},
            {"params": model.fc2.parameters(), "weight_decay": 0.001, "learning_rate": 0.1},
        ],
        weight_decay=0.01,
    )
    runner = Runner(datareader, model, optimizer)
    runner.run()
    expect = [
        26.45557023048071,
        26.189615890352265,
        25.925730483606507,
        25.66389296221932,
        25.404082443027733,
        25.14627820606084,
        24.89045969288372,
        24.636606504954457,
        24.384698401993962,
        24.134715300368555,
    ]
    runner.check(expect)
