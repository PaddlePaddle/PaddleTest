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


@pytest.mark.loss_margin_cross_entropy_vartype
def test_margin_cross_entropy_base():
    """
    test nn.functional.margin_cross_entropy base test
    """
    m1 = 1.0
    m2 = 0.5
    m3 = 0.0
    s = 64.0
    batch_size = 2
    # feature_length = 4
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
        reduction="mean",
    )
    runner.add_kwargs_to_dict("params_group3", dtype=np.float64, in_features=10, out_features=4)
    expect = [
        32.76620532659174,
        31.05348397348229,
        29.21074784865356,
        27.16523571208643,
        24.89044721455717,
        22.371238366928807,
        19.600606773582257,
        16.582370988701214,
        13.334508576034736,
        9.891639536137319,
    ]
    runner.run(model=Dygraph, expect=expect)
