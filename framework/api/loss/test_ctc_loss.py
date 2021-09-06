#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.functional.ctc_loss
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import numpy as np


@pytest.mark.loss_ctc_loss_vartype
def test_nn_cross_entropy_loss_base():
    """
    reduction='mean'
    """
    datareader = np.random.random(size=[5, 2, 10]) * 2 - 1
    label = np.array([[1, 2, 2], [1, 2, 2]]).astype("int32")
    loss = paddle.nn.functional.ctc_loss
    runner = Runner(datareader, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict(
        "params_group1",
        labels=label,
        input_lengths=np.array([5, 5]).astype("int64"),
        label_lengths=np.array([3, 3]).astype("int64"),
        blank=0,
        reduction="mean",
    )
    runner.add_kwargs_to_dict("params_group3", out_features=3)
    expect = [
        1.1823837647617452,
        1.182075016542177,
        1.1817664919368265,
        1.1814581907060235,
        1.1811501126103567,
        1.1808422574106694,
        1.180534624868062,
        1.1802272147438893,
        1.179920026799763,
        1.1796130607975504,
    ]
    runner.run(model=Dygraph, expect=expect)
