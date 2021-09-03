#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.KLDivLoss
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import numpy as np


@pytest.mark.loss_nn_KLDivLoss_vartype
def test_nn_kldiv_loss_base():
    """
    test nn.KLDivLoss base test
    """
    datareader = np.random.random(size=[5, 10]).astype("float64")
    label = np.random.random(size=[5, 2]).astype("float64")
    loss = paddle.nn.KLDivLoss
    runner = Runner(datareader, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict("params_group1", reduction="mean")
    runner.add_kwargs_to_dict("params_group2", label=label)
    expect = [
        -1.3249775194250017,
        -1.3252524655457654,
        -1.325527411666529,
        -1.325802357787293,
        -1.3260773039080567,
        -1.326352250028821,
        -1.3266271961495846,
        -1.3269021422703482,
        -1.3271770883911125,
        -1.327452034511876,
    ]
    runner.run(model=Dygraph, expect=expect)
