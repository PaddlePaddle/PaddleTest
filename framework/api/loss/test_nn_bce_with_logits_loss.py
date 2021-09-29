#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.BCEWithLogitsLoss
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import numpy as np


@pytest.mark.loss_nn_BCEWithLogitsLoss_vartype
def test_nn_bce_with_logits_loss_base():
    """
    test nn.BCEWithLogitsLoss base test
    """
    datareader = np.random.random(size=[5, 10])
    label = np.random.randint(0, 4, size=[5, 2]).astype(np.float64)
    loss = paddle.nn.BCEWithLogitsLoss
    runner = Runner(datareader, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict("params_group1", weight=None, reduction="mean", name=None)
    runner.add_kwargs_to_dict("params_group2", label=label)
    expect = [
        -3.3297434576122606,
        -3.3329866382553264,
        -3.3362291776935478,
        -3.339471077339924,
        -3.3427123386043447,
        -3.3459529628935885,
        -3.349192951611329,
        -3.3524323061581454,
        -3.3556710279315283,
        -3.358909118325883,
    ]
    runner.run(model=Dygraph, expect=expect)
