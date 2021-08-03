#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test Adadelta case
"""


import paddle
from runner import Runner
from base_dygraph_model import Dygraph
from linear_dygraph_model import LinearNet
from conv2d_dygraph_model import Conv2DNet
import reader


def test_adadelta_base():
    """
    test adadelta base test
    """
    model = Dygraph()
    datareader = reader.reader
    optimizer = paddle.optimizer.Adadelta(
        learning_rate=0.001,
        epsilon=1.0e-6,
        rho=0.95,
        parameters=model.parameters(),
        weight_decay=0.01,
        grad_clip=None,
        name=None,
    )
    runner = Runner(datareader, model, optimizer)
    runner.run()
    expect = [
        2.6125570230480717,
        2.5773767270777697,
        2.5466488417224844,
        2.5174922013728325,
        2.489115052044449,
        2.4611897092695147,
        2.4335504087732303,
        2.4061024049004347,
        2.378786869466334,
        2.351565015636534,
    ]
    runner.check(expect)


def test_adadelta_conv2d():
    """
    test adadelta opt with conv2d net test
    """
    model = Conv2DNet()
    datareader = reader.reader_img
    optimizer = paddle.optimizer.Adadelta(
        learning_rate=0.001,
        epsilon=1.0e-6,
        rho=0.95,
        parameters=model.parameters(),
        weight_decay=0.01,
        grad_clip=None,
        name=None,
    )
    runner = Runner(datareader, model, optimizer)
    runner.run()
    expect = [
        1.2335811384723961e-17,
        -0.006324175114315748,
        -0.011847891864908415,
        -0.017089408676644033,
        -0.02219110920714868,
        -0.02721191797670308,
        -0.032181642814653265,
        -0.03711732156261347,
        -0.04202953380878641,
        -0.0469252555457483,
    ]
    runner.check(expect)
