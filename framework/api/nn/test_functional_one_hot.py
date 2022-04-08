#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_one_hot
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestFunctionalOneHot(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestFunctionalOneHot(paddle.nn.functional.one_hot)


def cal_one_hot(x, num_classes):
    """
    calculate one_hot
    """
    res = []
    for i in x.flatten():
        tmp = [0] * num_classes
        tmp[i] = 1
        res.append(tmp)
    return np.array(res).reshape(list(x.shape) + [num_classes])


@pytest.mark.api_nn_one_hot_vartype
def test_one_hot_base():
    """
    base
    """
    x_data = np.random.randint(0, 6, (2, 3))
    res = cal_one_hot(x_data, 10)
    obj.base(res=res, x=x_data, num_classes=10)


@pytest.mark.api_nn_one_hot_parameters
def test_one_hot0():
    """
    default
    """
    x_data = np.random.randint(0, 6, (2, 3, 4, 6))
    res = cal_one_hot(x_data, 10)
    obj.run(res=res, x=x_data, num_classes=10)


@pytest.mark.skipif(paddle.is_compiled_with_cuda() is True, reason="skip cases because paddle is compiled with CUDA")
@pytest.mark.api_nn_one_hot_exception
def test_one_hot1():
    """
    num_classes < class
    """
    x_data = np.random.randint(0, 10, (10, 20))
    # res = cal_one_hot(x_data, 4)
    obj.exception(ValueError, mode="python", x=x_data, num_classes=4)


@pytest.mark.skipif(
    paddle.is_compiled_with_cuda() is not True, reason="skip cases because paddle is not compiled with CUDA"
)
@pytest.mark.api_nn_one_hot_exception
def test_one_hot2():
    """
    num_classes < class
    """
    x_data = np.random.randint(0, 10, (10, 20))
    obj.exception("CUDA error", mode="c", x=x_data, num_classes=4)
