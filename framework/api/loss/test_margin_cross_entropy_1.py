#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.functional.margin_cross_entropy
"""
import sys
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import numpy as np

sys.path.append("../..")
from utils.interceptor import skip_not_compile_gpu


@skip_not_compile_gpu
@pytest.mark.loss_margin_cross_entropy_parameters
def test_margin_cross_entropy_1():
    """
    test nn.functional.margin_cross_entropy test 1
    """
    m1 = 1.0
    m2 = 0.7
    m3 = 0.2
    s = 32.0
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
        reduction="mean",
    )
    runner.add_kwargs_to_dict("params_group3", dtype=np.float64, in_features=10, out_features=4)
    expect = [
        29.340229533853694,
        28.95557148505032,
        28.554827050190063,
        28.13519789563582,
        27.69420186026433,
        27.22964547817405,
        26.739585472118474,
        26.22229815120594,
        25.67626479019914,
        25.100174834480327,
    ]
    runner.run(model=Dygraph, expect=expect)
