#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test Adadelta case
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
from linear_dygraph_model import LinearNet
from conv2d_dygraph_model import Conv2DNet
import reader


@pytest.mark.api_optimizer_adadelta_vartype
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
        2.6125321429832447,
        2.6125069460652783,
        2.6124815347999717,
        2.6124559599406045,
        2.6124302517130675,
        2.6124044301398666,
        2.6123785094395275,
        2.612352500216587,
        2.612326410673878,
    ]
    runner.check(expect)


@pytest.mark.api_optimizer_adadelta_parameters
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
        -6.938893903907228e-18,
        -4.471956541560185e-06,
        -9.000880473161721e-06,
        -1.3568339349880054e-05,
        -1.816520788137238e-05,
        -2.278605188502092e-05,
        -2.742727174084164e-05,
        -3.2086311329657113e-05,
        -3.676126426863796e-05,
        -4.1450655952610105e-05,
    ]
    runner.check(expect)
