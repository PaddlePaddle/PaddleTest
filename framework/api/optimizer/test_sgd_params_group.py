#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test sgd support params groups case
"""


import paddle
from runner import Runner
from linear_dygraph_model import LinearNet
import reader


def test_sgd_params_group():
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
    # expect = [26.45557023,
    #           26.38481750,
    #           26.24340570,
    #           26.03141642,
    #           25.74897134,
    #           25.39623154,
    #           24.97339663,
    #           24.48070370,
    #           23.91842595,
    #           23.28687119]
    expect = [
        26.45557023,
        26.38481750,
        26.31408541,
        26.24337390,
        26.17268292,
        26.10201242,
        26.03136234,
        25.96073263,
        25.89012322,
        25.81953407,
    ]
    runner.check(expect)
