#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.MarginRankingLoss
"""

import paddle
from runner import Runner
from base_dygraph_model import Dygraph
import reader
import numpy as np


def test_nn_margin_ranking_loss_base():
    """
    test nn.MarginRankingLoss base test
    """
    model = Dygraph()
    datareader = np.random.random(size=[2, 10])
    label = paddle.to_tensor([[1, -1], [-1, -1]], dtype="float64")
    other = np.array([[2, 1], [2, 4]]).astype(np.float64)
    loss = paddle.nn.MarginRankingLoss
    runner = Runner(datareader, other, model, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict("params_group1", margin=0.0, reduction="mean", name=None)
    runner.add_kwargs_to_dict("params_group2", label=label)
    runner.run()
    expect = [
        0.7593716739231385,
        0.7587214473614408,
        0.7580712207997423,
        0.7574209942380448,
        0.7567707676763465,
        0.7561205411146485,
        0.7554703145529503,
        0.7548200879912526,
        0.7541698614295543,
        0.7535196348678563,
    ]
    runner.check(expect)
