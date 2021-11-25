#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test Lamb case
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
from linear_dygraph_model import LinearNet
import reader


@pytest.mark.api_optimizer_lamb_vartype
def test_lamb_base():
    """
    test lamb base test
    """
    model = Dygraph()
    datareader = reader.reader
    optimizer = paddle.optimizer.Lamb(
        learning_rate=0.001,
        lamb_weight_decay=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-06,
        parameters=model.parameters(),
        grad_clip=None,
        exclude_from_weight_decay_fn=None,
    )
    runner = Runner(datareader, model, optimizer)
    runner.run()
    expect = [
        2.6125570230480717,
        2.6099444421328157,
        2.6073344737985504,
        2.6047271154326452,
        2.6021223644251172,
        2.599520218168603,
        2.5969206740583504,
        2.59432372949221,
        2.591729381870638,
        2.589137628596689,
    ]
    runner.check(expect)
