#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.CrossEntropyLoss
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import numpy as np


@pytest.mark.loss_nn_CrossEntropyLoss_parameters
def test_nn_cross_entropy_loss_2():
    """
    test nn.CrossEntropyLoss test 2
    """
    datareader = np.random.random(size=[5, 20])
    label = np.random.randint(0, 2, size=(5)).astype(np.int64)
    label[0] = -25
    ignore_index = -25
    # provide weight for normal label and ignore label with ignore_index
    weight = np.array([0.2, 0.3, 0.1, 0.5, 0.4, 0.7, 0.2, 0.77, 0.45, 0.8])
    loss = paddle.nn.CrossEntropyLoss
    runner = Runner(datareader, loss)
    runner.case_name = "_case1"
    runner.softmax = False
    runner.add_kwargs_to_dict(
        "params_group1",
        weight=weight,
        ignore_index=ignore_index,
        reduction="mean",
        soft_label=False,
        axis=-1,
        name=None,
    )
    runner.add_kwargs_to_dict("params_group2", label=label)
    runner.add_kwargs_to_dict("params_group3", dtype=np.float64, in_features=20, out_features=10)
    expect = [
        2.3025850929940463,
        2.299455474047083,
        2.2963297893271752,
        2.2932080445254703,
        2.2900902453078524,
        2.286976397314741,
        2.283866506160874,
        2.2807605774351107,
        2.2776586167002213,
        2.274560629492685,
    ]
    runner.run(model=Dygraph, expect=expect)
