#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.functional.mse_loss
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import numpy as np


@pytest.mark.loss_mse_loss_vartype
def test_nn_mse_loss_base():
    """
    test nn.functional.mse_loss base test
    """
    datareader = np.random.random(size=[5, 10])
    label = np.random.randint(0, 1, size=[5, 2]).astype(np.float64)
    loss = paddle.nn.functional.mse_loss
    runner = Runner(datareader, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict("params_group1", label=label, reduction="mean", name=None)
    expect = [
        7.019458764884789,
        6.973347883852658,
        6.92754050576595,
        6.88203463264913,
        6.836828279679613,
        6.791919475101177,
        6.747306260137942,
        6.702986688908922,
        6.658958828343143,
        6.615220758095295,
    ]
    runner.run(model=Dygraph, expect=expect)
