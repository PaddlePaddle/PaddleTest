#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.functional.dice_loss
"""

import paddle
from runner import Runner
from base_dygraph_model import Dygraph
import numpy as np


def test_dice_loss_base():
    """
    test dice_loss base test
    """
    datareader = np.random.randn(5, 10)
    label = np.random.randint(0, 2, size=(5, 1))
    loss = paddle.nn.functional.dice_loss
    runner = Runner(datareader, loss)
    runner.softmax = True
    runner.add_kwargs_to_dict("params_group1", label=label, epsilon=1e-05)
    expect = [
        0.5000024999874368,
        0.4998159456312921,
        0.4996293913964941,
        0.4994428374448372,
        0.49925628393811544,
        0.49906973103811947,
        0.4988831789066369,
        0.49869662770545126,
        0.49851007759634064,
        0.4983235287410773,
    ]
    runner.run(model=Dygraph, expect=expect)
