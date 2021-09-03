#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.functional.binary_cross_entropy_with_logits
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import reader
import numpy as np


@pytest.mark.loss_binary_cross_entropy_with_logits_vartype
def test_binary_cross_entropy_with_logits_base():
    """
    test binary_cross_entropy_with_logits base test
    """
    datareader = reader.reader
    label = np.array([[[0, 1]]]).astype(np.float64)
    loss = paddle.nn.functional.binary_cross_entropy_with_logits
    runner = Runner(datareader, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict("params_group1", label=label, weight=None, reduction="mean", name=None)
    expect = [
        1.3770600864641613,
        1.376208948137482,
        1.3753580214612544,
        1.3745073067130529,
        1.3736568041706076,
        1.3728065141118022,
        1.3719564368146735,
        1.3711065725574116,
        1.3702569216183584,
        1.369407484276008,
    ]
    runner.run(model=Dygraph, expect=expect)
    # runner.check(expect)
