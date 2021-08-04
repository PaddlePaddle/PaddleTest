#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test RMSProp case
"""


import paddle
from runner import Runner
from base_dygraph_model import Dygraph
from linear_dygraph_model import LinearNet
import reader


def test_rmsprop_base():
    """
    test rmsprop base test
    """
    model = Dygraph()
    datareader = reader.reader
    optimizer = paddle.optimizer.RMSProp(
        learning_rate=0.001,
        rho=0.95,
        epsilon=1e-06,
        momentum=0.0,
        centered=False,
        parameters=model.parameters(),
        weight_decay=None,
        grad_clip=None,
        name=None,
    )
    runner = Runner(datareader, model, optimizer)
    runner.run()
    expect = [
        2.6125570230480717,
        2.587681377682089,
        2.569863466410184,
        2.5551302890912857,
        2.542210734049388,
        2.530511471542003,
        2.519700051053436,
        2.509568526409088,
        2.4999768246556466,
        2.4908254116504214,
    ]
    runner.check(expect)
