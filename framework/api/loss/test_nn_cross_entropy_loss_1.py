#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.CrossEntropyLoss
"""
import paddle
import pytest
from runner import Runner
from conv2d_dygraph_model import Conv2DNet
import numpy as np


@pytest.mark.loss_nn_CrossEntropyLoss_parameters
def test_nn_cross_entropy_loss_1():
    """
    test nn.CrossEntropyLoss base test
    """
    datareader = np.random.random(size=[4, 125, 125, 7])
    label = np.random.randint(0, 7, size=(4, 125, 125)).astype(np.int64)
    label[0, 0, 0] = 255
    weight = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    loss = paddle.nn.CrossEntropyLoss
    runner = Runner(datareader, loss)
    runner.case_name = "_case1"
    runner.softmax = False
    runner.add_kwargs_to_dict(
        "params_group1", weight=weight, ignore_index=255, reduction="mean", soft_label=False, axis=-1, name=None
    )
    runner.add_kwargs_to_dict("params_group2", label=label)
    runner.add_kwargs_to_dict("params_group3", dtype=np.float64, in_channels=7, out_channels=7, data_format="NHWC")
    expect = [
        1.9459101490553212,
        1.9458743456438323,
        1.945838552450792,
        1.9458027694732092,
        1.9457669967082154,
        1.945731234152911,
        1.9456954818044199,
        1.9456597396597235,
        1.9456240077159956,
        1.9455882859703635,
    ]
    runner.run(model=Conv2DNet, expect=expect)
