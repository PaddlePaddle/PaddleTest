#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.functional.kl_div
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import numpy as np


@pytest.mark.loss_kl_div_vartype
def test_kl_div_base():
    """
    test nn.functional.kl_div base test
    """
    datareader = np.random.random(size=[5, 10]).astype("float32")
    label = np.random.random(size=[5, 2]).astype("float32")
    loss = paddle.nn.functional.kl_div
    runner = Runner(datareader, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict("params_group1", label=label, reduction="mean", name=None)
    expect = [
        -1.3249776,
        -1.3252525,
        -1.3255274,
        -1.3258024,
        -1.3260773,
        -1.3263524,
        -1.3266273,
        -1.3269022,
        -1.3271772,
        -1.3274521,
    ]
    runner.run(model=Dygraph, expect=expect, dtype=np.float32)
