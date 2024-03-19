#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test Adagrad case
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
from linear_dygraph_model import LinearNet
from conv2d_dygraph_model import Conv2DNet
import reader


@pytest.mark.api_optimizer_adagrad_vartype
def test_adagrad_base():
    """
    test adagrad base test
    """
    model = Dygraph()
    datareader = reader.reader
    optimizer = paddle.optimizer.Adagrad(
        learning_rate=0.001,
        epsilon=1.0e-6,
        parameters=model.parameters(),
        weight_decay=None,
        grad_clip=None,
        name=None,
        initial_accumulator_value=0.0,
    )
    runner = Runner(datareader, model, optimizer)
    runner.run()
    expect = [
        2.6125570230480717,
        2.6069919310016734,
        2.6030568121214954,
        2.5998437993621866,
        2.5970612478391,
        2.5945724575771516,
        2.592300512948242,
        2.5901971006934144,
        2.5882295385033647,
        2.5863745029324328,
    ]
    runner.check(expect)


@pytest.mark.api_optimizer_adagrad_parameters
def test_adagrad_conv2d():
    """
    test adagrad opt with conv2d net test
    """
    model = Conv2DNet()
    datareader = reader.reader_img
    optimizer = paddle.optimizer.Adagrad(
        learning_rate=0.001,
        epsilon=1.0e-6,
        parameters=model.parameters(),
        weight_decay=None,
        grad_clip=None,
        name=None,
        initial_accumulator_value=0.0,
    )
    runner = Runner(datareader, model, optimizer)
    runner.run()
    expect = [
        1.2335811384723961e-17,
        -0.000999998000003974,
        -0.0017071037811919815,
        -0.0022844533837157333,
        -0.0027844528837161855,
        -0.0032316660792165194,
        -0.0036399140363473667,
        -0.004017878223642499,
        -0.004371431364235956,
        -0.0047047644753471785,
    ]
    runner.check(expect)
