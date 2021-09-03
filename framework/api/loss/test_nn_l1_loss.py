#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.L1Loss
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import numpy as np


@pytest.mark.loss_nn_L1Loss_vartype
def test_nn_l1_loss_base():
    """
    test nn.L1Loss base test
    """
    datareader = np.random.random(size=[2, 10])
    label = np.array([[1.7, 1.0], [0.4, 0.5]]).astype(np.float64)
    loss = paddle.nn.L1Loss
    runner = Runner(datareader, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict("params_group1", reduction="mean", name=None)
    runner.add_kwargs_to_dict("params_group2", label=label)
    expect = [
        1.8620622352193652,
        1.8602888649315787,
        1.8585154946437927,
        1.8567421243560063,
        1.8549687540682194,
        1.8531953837804336,
        1.851422013492647,
        1.849648643204861,
        1.8478752729170744,
        1.8461019026292877,
    ]
    runner.run(model=Dygraph, expect=expect)
