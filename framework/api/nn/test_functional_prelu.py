#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_prelu
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestPrelu(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.delta = 1e-3
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestPrelu(paddle.nn.functional.prelu)


@pytest.mark.api_nn_prelu_vartype
def test_prelu_base():
    """
    base
    """
    data = np.array(
        [
            [
                [[-2.0, 3.0, -4.0, 5.0], [3.0, -4.0, 5.0, -6.0], [-7.0, -8.0, 8.0, 9.0]],
                [[1.0, -2.0, -3.0, 4.0], [-5.0, 6.0, 7.0, -8.0], [6.0, 7.0, 8.0, 9.0]],
            ]
        ]
    )
    # num_parameters = 1
    weight = np.array([0.25])
    res = np.array(
        [
            [
                [[-0.5, 3.0, -1.0, 5.0], [3.0, -1.0, 5.0, -1.5], [-1.75, -2.0, 8.0, 9.0]],
                [[1.0, -0.5, -0.75, 4.0], [-1.25, 6.0, 7.0, -2.0], [6.0, 7.0, 8.0, 9.0]],
            ]
        ]
    )
    obj.base(res=res, x=data, weight=weight)


@pytest.mark.api_nn_prelu_parameters
def test_prelu():
    """
    default
    """
    data = np.array(
        [
            [
                [[-2.0, 3.0, -4.0, 5.0], [3.0, -4.0, 5.0, -6.0], [-7.0, -8.0, 8.0, 9.0]],
                [[1.0, -2.0, -3.0, 4.0], [-5.0, 6.0, 7.0, -8.0], [6.0, 7.0, 8.0, 9.0]],
            ]
        ]
    )
    # num_parameters = 1
    weight = np.array([0.25])
    res = np.array(
        [
            [
                [[-0.5, 3.0, -1.0, 5.0], [3.0, -1.0, 5.0, -1.5], [-1.75, -2.0, 8.0, 9.0]],
                [[1.0, -0.5, -0.75, 4.0], [-1.25, 6.0, 7.0, -2.0], [6.0, 7.0, 8.0, 9.0]],
            ]
        ]
    )
    obj.run(res=res, x=data, weight=weight)


@pytest.mark.api_nn_prelu_parameters
def test_prelu1():
    """
    weight = [in]
    """
    data = randtool("float", -10, 10, [3, 3, 3])
    weight = np.array([0.5, 0.5, 0.5])
    res = np.maximum(data, 0) + weight * np.minimum(data, 0)
    obj.run(res=res, x=data, weight=weight)
