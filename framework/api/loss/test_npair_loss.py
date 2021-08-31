#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.functional.npair_loss
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import numpy as np


@pytest.mark.loss_npair_loss_vartype
def test_npair_loss_base():
    """
    test npair_loss base test
    """
    datareader = np.random.random(size=[18, 10]).astype(np.float32)
    positive = np.random.random(size=[18, 6]).astype(np.float32)
    label = np.random.random(size=[18]).astype(np.float32)
    loss = paddle.nn.functional.npair_loss
    runner = Runner(datareader, loss)
    runner.softmax = False
    runner.add_kwargs_to_dict("params_group1", positive=positive, labels=label, l2_reg=0.002)
    expect = [4.1371465, 4.1366587, 4.136171, 4.135683, 4.1351953, 4.134708, 4.1342216, 4.1337347, 4.1332483, 4.1327624]
    runner.run(model=Dygraph, expect=expect, dtype=np.float32, out_features=6)
