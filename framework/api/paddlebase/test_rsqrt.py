#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test rsqrt
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestRsqrt(APIBase):
    """
    test rsqrt
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = True


obj = TestRsqrt(paddle.rsqrt)


@pytest.mark.api_base_rsqrt_vartype
def test_rsqrt_base():
    """
    rsqrt_base
    """
    x_data = np.arange(1, 13).reshape((2, 3, 2)).astype(np.float32)
    res = np.array(
        [
            [[1.0, 0.7071067], [0.57735026, 0.5], [0.44721356, 0.40824828]],
            [[0.37796447, 0.35355335], [0.3333333, 0.31622776], [0.30151135, 0.28867513]],
        ]
    )
    obj.base(res=res, x=x_data)


@pytest.mark.api_base_rsqrt_parameters
def test_rsqrt_input0():
    """
    input=[1, 13]
    """
    x_data = np.arange(1, 13).reshape((2, 3, 2)).astype(np.float32)
    res = np.array(
        [
            [[1.0, 0.7071067], [0.57735026, 0.5], [0.44721356, 0.40824828]],
            [[0.37796447, 0.35355335], [0.3333333, 0.31622776], [0.30151135, 0.28867513]],
        ]
    )
    obj.base(res=res, x=x_data)


@pytest.mark.api_base_rsqrt_parameters
def test_rsqrt_input1():
    """
    input = [-3, 15]
    """
    obj.enable_backward = False
    x_data = np.arange(-3, 15).reshape((2, 3, 3)).astype(np.float32)
    res = np.array(
        [
            [[np.nan, np.nan, np.nan], [np.inf, 1.0, 0.7071067], [0.57735026, 0.5, 0.44721356]],
            [
                [0.40824828, 0.37796447, 0.35355335],
                [0.3333333, 0.31622776, 0.30151135],
                [0.28867513, 0.2773501, 0.26726124],
            ],
        ]
    )
    obj.run(res=res, x=x_data)


@pytest.mark.api_base_rsqrt_parameters
def test_rsqrt_input2():
    """
    input = 0d array
    """
    obj.enable_backward = False
    x_data = np.array(2, dtype=np.float32)
    res = np.array(0.7071067690849304)
    obj.run(res=res, x=x_data)
