#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test Momentum case
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
from linear_dygraph_model import LinearNet
import reader


@pytest.mark.api_optimizer_momentum_vartype
def test_momentum_base():
    """
    test momentum base test
    """
    model = Dygraph()
    datareader = reader.reader
    optimizer = paddle.optimizer.Momentum(
        learning_rate=0.001,
        momentum=0.9,
        parameters=model.parameters(),
        use_nesterov=False,
        weight_decay=None,
        grad_clip=None,
        name=None,
    )
    runner = Runner(datareader, model, optimizer)
    runner.run()
    expect = [
        2.6125570230480717,
        2.6106062484608996,
        2.6068997767917828,
        2.6016131777907745,
        2.5949044642287373,
        2.58691584759568,
        2.57777531822922,
        2.5675980674301604,
        2.55648776736648,
        2.5445377229868855,
    ]
    runner.check(expect)
