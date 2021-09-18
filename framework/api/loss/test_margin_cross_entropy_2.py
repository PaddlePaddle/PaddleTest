#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.functional.margin_cross_entropy
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import numpy as np


@pytest.mark.loss_margin_cross_entropy_parameters
def test_margin_cross_entropy_2():
    """
    test nn.functional.margin_cross_entropy test 2
    """
    m1 = 0.8
    m2 = 0.0
    m3 = 0.15
    s = 40.0
    batch_size = 2
    feature_length = 10
    num_classes = 4
    datareader = np.random.randn(batch_size, feature_length).astype("float64")
    label = np.random.randint(0, high=num_classes, size=(batch_size)).astype("int64")
    loss = paddle.nn.functional.margin_cross_entropy
    runner = Runner(datareader, loss)
    runner.softmax = True
    runner.mean_loss = True
    runner.add_kwargs_to_dict(
        "params_group1",
        label=label,
        margin1=m1,
        margin2=m2,
        margin3=m3,
        scale=s,
        group=None,
        return_softmax=False,
        reduction=None,
    )
    runner.add_kwargs_to_dict("params_group3", dtype=np.float64, in_features=10, out_features=4)
    expect = [
        0.06838991078472811,
        0.06661214136650885,
        0.0649204641090921,
        0.06330883999857849,
        0.061771780397832654,
        0.060304285789012105,
        0.05890179251431488,
        0.057560126323949176,
        0.05627546173840599,
        0.05504428639402802,
    ]
    runner.run(model=Dygraph, expect=expect)
