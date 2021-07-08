#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test sgd case
"""


import paddle
from runner import Runner
from base_dygraph_model import Dygraph
import reader


def test_sgd():
    """
    test sgd base test
    """
    model = Dygraph()
    datareader = reader.reader
    optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    runner = Runner(datareader, model, optimizer)
    # runner.debug = True
    runner.run()
    expect = [
        0.5172499461644406,
        0.5152991715772685,
        0.5113976224029243,
        0.505545298641408,
        0.4977422002927195,
        0.48798832735685893,
        0.47628367983382636,
        0.4626282577236215,
        0.44702206102624464,
        0.4294650897416956,
    ]
    runner.check(expect)
