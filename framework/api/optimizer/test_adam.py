#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test Adam case
"""


import paddle
from runner import Runner
from base_dygraph_model import Dygraph
from linear_dygraph_model import LinearNet
import reader


def test_adam_base():
    """
    test adam base test
    """
    model = Dygraph()
    datareader = reader.reader
    optimizer = paddle.optimizer.Adam(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        parameters=model.parameters(),
        weight_decay=None,
        grad_clip=None,
        name=None,
        lazy_mode=False,
    )
    runner = Runner(datareader, model, optimizer)
    runner.run()
    expect = [
        2.6125570230480717,
        2.6069918720704797,
        2.601426739712486,
        2.5958616135601766,
        2.590296490510068,
        2.5847313693207723,
        2.579166249371597,
        2.573601130307867,
        2.5680360119079126,
        2.5624708940239613,
    ]
    runner.check(expect)
