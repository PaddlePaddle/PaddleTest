#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.functional.l1_loss
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import numpy as np


@pytest.mark.loss_l1_loss_vartype
def test_l1_loss_base():
    """
    test l1_loss base test
    """
    datareader = np.random.random(size=[2, 10])
    label = np.random.random(size=[2, 2]).astype(np.float64)
    loss = paddle.nn.functional.l1_loss
    runner = Runner(datareader, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict("params_group1", label=label, reduction="mean", name=None)
    expect = [
        2.32859375991682,
        2.3268203896290327,
        2.3250470193412474,
        2.3232736490534602,
        2.321500278765674,
        2.3197269084778878,
        2.3179535381901015,
        2.3161801679023153,
        2.314406797614529,
        2.312633427326742,
    ]
    runner.run(model=Dygraph, expect=expect)
