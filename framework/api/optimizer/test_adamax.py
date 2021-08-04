#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test Adamax case
"""


import paddle
from runner import Runner
from base_dygraph_model import Dygraph
from linear_dygraph_model import LinearNet
import reader


def test_adamax_base():
    """
    test adamax base test
    """
    model = Dygraph()
    datareader = reader.reader
    optimizer = paddle.optimizer.Adamax(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        parameters=model.parameters(),
        weight_decay=None,
        grad_clip=None,
        name=None,
    )
    runner = Runner(datareader, model, optimizer)
    runner.run()
    expect = [
        2.6125570230480717,
        2.606991907675149,
        2.6014267930005563,
        2.595861678557881,
        2.590296564230524,
        2.5847314499718492,
        2.5791663357585426,
        2.5736012215772877,
        2.568036107419764,
        2.5624709932804293,
    ]
    runner.check(expect)
