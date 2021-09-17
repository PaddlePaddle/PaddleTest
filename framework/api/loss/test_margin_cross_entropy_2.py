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
        0.06498025172082632,
        0.0618818588414974,
        0.059054196107834404,
        0.05646349220212549,
        0.054081379514777886,
        0.05188385041399612,
        0.0498504474459089,
        0.047963628289570535,
        0.04620826284448239,
    ]
    runner.run(model=Dygraph, expect=expect)
