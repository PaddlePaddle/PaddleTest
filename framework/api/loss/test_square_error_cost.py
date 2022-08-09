#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.functional.square_error_cost
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import reader
import numpy as np


@pytest.mark.loss_square_error_cost_vartype
def test_square_error_cost_base():
    """
    test square_error_cost base test
    """
    datareader = np.random.random(size=[3, 2, 1, 10]).astype(np.float64)
    label = np.random.random(size=[3, 2, 1, 2]).astype(np.float64)
    loss = paddle.nn.functional.square_error_cost
    runner = Runner(datareader, loss)
    runner.mean_loss = True
    runner.softmax = False
    runner.add_kwargs_to_dict("params_group1", label=label)
    runner.add_kwargs_to_dict("params_group3", dtype=np.float64, in_features=10, out_features=2)
    expect = [
        5.274356995458327,
        5.236291188929251,
        5.198503988905537,
        5.160993355078503,
        5.123757262081596,
        5.086793699380965,
        5.050100671166836,
        5.0136761962456795,
        4.977518307933168,
        4.941625053947916,
    ]
    runner.run(model=Dygraph, expect=expect)
