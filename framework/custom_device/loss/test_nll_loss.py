#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.functional.nll_loss
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import numpy as np


@pytest.mark.loss_nll_loss_vartype
def test_nll_loss_base():
    """
    test nll_loss base test
    """
    datareader = np.random.random(size=[5, 10])
    label = np.random.randint(0, 1, size=[5, 5]).astype(np.int64)
    loss = paddle.nn.functional.nll_loss
    runner = Runner(datareader, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict("params_group1", label=label, weight=None, ignore_index=-100, reduction="mean", name=None)
    expect = [
        -2.642447338415548,
        -2.6457262284340075,
        -2.649005118452467,
        -2.6522840084709265,
        -2.6555628984893858,
        -2.658841788507845,
        -2.6621206785263043,
        -2.6653995685447645,
        -2.6686784585632237,
        -2.6719573485816834,
    ]
    runner.run(model=Dygraph, expect=expect)
