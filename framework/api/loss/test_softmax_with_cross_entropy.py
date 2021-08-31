#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.functional.softmax_with_cross_entropy
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import numpy as np


@pytest.mark.loss_softmax_with_cross_entropy_vartype
def test_softmax_with_cross_entropy_base():
    """
    test softmax_with_cross_entropy base test
    """
    datareader = np.random.random(size=[5, 10]).astype(np.float32)
    label = np.array([[1], [0], [1], [1], [0]]).astype(np.int64)
    loss = paddle.nn.functional.softmax_with_cross_entropy
    runner = Runner(datareader, loss)
    runner.static = False
    runner.softmax = False
    runner.add_kwargs_to_dict(
        "params_group1",
        label=label,
        soft_label=False,
        ignore_index=-100,
        numeric_stable_mode=True,
        return_softmax=False,
        axis=-1,
    )
    expect = [
        0.6931472,
        0.69111603,
        0.6891015,
        0.6871032,
        0.68512106,
        0.6831546,
        0.681204,
        0.679269,
        0.6773496,
        0.6754453,
    ]
    runner.run(model=Dygraph, expect=expect, dtype=np.float32)
