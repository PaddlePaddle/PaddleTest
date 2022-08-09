#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_HingeEmbeddingLoss
"""
import copy

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


def cal_HingeEmbeddingLoss(x, labels, **kwargs):
    """
    class convert function
    """
    hinge_embedding_loss = paddle.nn.HingeEmbeddingLoss(**kwargs)
    return hinge_embedding_loss(x, labels)


def cal_np(input, labels, margin=1.0, reduction="mean"):
    """
    calculate api with np
    """
    x = copy.deepcopy(input)
    x[labels == -1] = np.where(margin - x[labels == -1] > 0, margin - x[labels == -1], 0)
    # print(x)
    if reduction == "mean":
        return np.mean(x).reshape(1)
    elif reduction == "none":
        return x
    elif reduction == "sum":
        return np.sum(x).reshape(1)


class TestHingeEmbeddingLoss(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = False
        # enable check grad
        self.enable_backward = False
        # self.delta = 1e-3


obj = TestHingeEmbeddingLoss(cal_HingeEmbeddingLoss)


@pytest.mark.api_nn_HingeEmbeddingLoss_vartype
def test_HingeEmbeddingLoss_base():
    """
    base
    """
    x = np.array([[1, -2, 3], [0, -1, 2], [1, 0, 1]], dtype="float32")
    label = np.array([[-1, 1, -1], [1, 1, 1], [1, -1, 1]])
    res = cal_np(x, label, reduction="none")
    obj.base(res=res, x=x, labels=label, reduction="none")


@pytest.mark.api_nn_HingeEmbeddingLoss_vartype
def test_HingeEmbeddingLoss0():
    """
    default
    """
    x = randtool("float", -4, 5, (3, 3))
    label = np.array([[-1, 1, -1], [1, 1, 1], [1, -1, 1]])
    res = cal_np(x, label)
    obj.run(res=res, x=x, labels=label)


@pytest.mark.api_nn_HingeEmbeddingLoss_vartype
def test_HingeEmbeddingLoss1():
    """
    margin=4.0
    """
    x = randtool("float", -4, 5, (3, 3))
    label = np.array([[-1, 1, -1], [1, 1, 1], [1, -1, 1]])
    res = cal_np(x, label, margin=4.0)
    obj.run(res=res, x=x, labels=label, margin=4.0)


@pytest.mark.api_nn_HingeEmbeddingLoss_vartype
def test_HingeEmbeddingLoss2():
    """
    margin=-4.0
    """
    x = randtool("float", -4, 5, (3, 3))
    label = np.array([[-1, 1, -1], [1, 1, 1], [1, -1, 1]])
    res = cal_np(x, label, margin=-4.0)
    obj.run(res=res, x=x, labels=label, margin=-4.0)


@pytest.mark.api_nn_HingeEmbeddingLoss_vartype
def test_HingeEmbeddingLoss3():
    """
    x: 3-d tensor
    margin=-4.0
    """
    x = randtool("float", -4, 5, (4, 3, 3))
    label = np.ones((4, 3, 3))
    res = cal_np(x, label, margin=-4.0)
    obj.run(res=res, x=x, labels=label, margin=-4.0)


@pytest.mark.api_nn_HingeEmbeddingLoss_vartype
def test_HingeEmbeddingLoss4():
    """
    x: 4-d tensor
    margin=-4.0
    """
    x = randtool("float", -4, 5, (4, 3, 3, 4))
    label = -np.ones((4, 3, 3, 4))
    res = cal_np(x, label, margin=-4.0)
    obj.run(res=res, x=x, labels=label, margin=-4.0)


@pytest.mark.api_nn_HingeEmbeddingLoss_vartype
def test_HingeEmbeddingLoss5():
    """
    x: 4-d tensor
    margin=-4.0
    reduction='none'
    """
    x = randtool("float", -4, 5, (4, 3, 3, 4))
    label = -np.ones((4, 3, 3, 4))
    res = cal_np(x, label, margin=-4.0, reduction="none")
    obj.run(res=res, x=x, labels=label, margin=-4.0, reduction="none")


@pytest.mark.api_nn_HingeEmbeddingLoss_vartype
def test_HingeEmbeddingLoss6():
    """
    x: 4-d tensor
    margin=-4.0
    reduction='sum'
    """
    x = randtool("float", -4, 5, (4, 3, 3, 4))
    label = -np.ones((4, 3, 3, 4))
    res = cal_np(x, label, margin=-4.0, reduction="sum")
    obj.run(res=res, x=x, labels=label, margin=-4.0, reduction="sum")
