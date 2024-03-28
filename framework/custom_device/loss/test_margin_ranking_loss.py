#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.functional.margin_ranking_loss
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import numpy as np


@pytest.mark.loss_margin_ranking_loss_vartype
def test_margin_ranking_loss_base():
    """
    test nn.functional.margin_ranking_loss base test
    """
    datareader = np.random.random(size=[2, 10])
    label = np.array([[1, -1], [-1, -1]]).astype(np.float64)
    other = np.array([[2, 1], [2, 4]]).astype(np.float64)
    loss = paddle.nn.functional.margin_ranking_loss
    runner = Runner(datareader, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict("params_group1", other=other, label=label, margin=0.0, reduction="mean", name=None)
    expect = [
        0.6310311176096826,
        0.6305001903870682,
        0.6299692631644536,
        0.6294383359418392,
        0.6289074087192248,
        0.6283764814966103,
        0.6278455542739958,
        0.6273146270513813,
        0.6267836998287669,
        0.6262527726061524,
    ]
    runner.run(model=Dygraph, expect=expect)
