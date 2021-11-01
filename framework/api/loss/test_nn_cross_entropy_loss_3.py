#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.CrossEntropyLoss
"""
import paddle
import pytest
import numpy as np


@pytest.mark.loss_nn_CrossEntropyLoss_vartype
def test_nn_cross_entropy_loss_3():
    """
    test nn.CrossEntropyLoss 3 test
    """
    np.random.seed(33)
    datareader = np.random.random(size=[25, 529, 44, 44])
    label = np.random.randint(0, 529, size=(25, 1, 44, 44)).astype(np.int64)
    exp = np.array([6.31244271]).astype(np.float32)
    layer = paddle.nn.CrossEntropyLoss(axis=1)
    res = layer(paddle.to_tensor(datareader), paddle.to_tensor(label))
    assert np.allclose(res.numpy(), exp)
