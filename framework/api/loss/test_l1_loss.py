#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.functional.l1_loss
"""

import paddle
from runner import Runner
from base_dygraph_model import Dygraph
import reader
import numpy as np


def test_l1_loss_base():
    """
    test l1_loss base test
    """
    model = Dygraph()
    datareader = np.random.random(size=[2, 10])
    label = np.random.random(size=[2, 2]).astype(np.float64)
    loss = paddle.nn.functional.l1_loss
    runner = Runner(datareader, label, model, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict("params_group1", reduction="mean", name=None)
    runner.run()
    expect = [
        2.5220001489695374,
        2.519709608944252,
        2.5174190689189673,
        2.5151285288936824,
        2.5128379888683967,
        2.510547448843112,
        2.5082569088178266,
        2.5059663687925413,
        2.503675828767257,
        2.5013852887419707,
    ]
    runner.check(expect)
