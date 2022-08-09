#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test mm
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestMm(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestMm(paddle.mm)


@pytest.mark.api_base_mm_vartype
def test_mm_base():
    """
    base
    """
    x = np.array([[1.0, 2.1, 2.0], [3.1, 4.4, 1.2]])
    mat2 = np.array([[2.0, 3.0], [2.1, 1.5], [2.5, 6.1]])
    res = np.dot(x, mat2)
    obj.base(res=res, input=x, mat2=mat2)


@pytest.mark.api_base_mm_parameters
def test_mm1():
    """
    input: [B, ..., M, K], mat2: [B, ..., K, N]
    """
    x = randtool("float", -10, 10, [2, 3, 1, 2])
    mat2 = randtool("float", -10, 10, [2, 3, 2, 3])
    res = np.array(
        [
            [
                [[32.62671841, 7.33150530, 33.95572904]],
                [[27.71220426, -10.03697974, 26.41360584]],
                [[11.69667616, -74.58606618, 39.09153314]],
            ],
            [
                [[-120.91272325, -88.25803251, -81.61568465]],
                [[2.20863882, -23.38574287, -13.92580615]],
                [[29.26243762, -98.75744044, 5.89729943]],
            ],
        ]
    )
    obj.run(res=res, input=x, mat2=mat2)


@pytest.mark.api_base_mm_parameters
def test_mm2():
    """
    input: [B, M, K], mat2: [B, K, N]
    """
    x = randtool("float", -10, 10, [2, 3, 4])
    mat2 = randtool("float", -10, 10, [2, 4, 2])
    res = np.array(
        [
            [[-18.61078501, -53.60577087], [81.96771615, 4.20841618], [43.85200283, -10.86080817]],
            [[17.99607183, -36.79511343], [-9.74683385, 15.04989944], [50.3349109, -94.68744424]],
        ]
    )
    obj.run(res=res, input=x, mat2=mat2)


@pytest.mark.api_base_mm_parameters
def test_mm3():
    """
    input: [B, M, K], mat2: [K, N]
    """
    x = randtool("float", -10, 10, [2, 3, 4])
    mat2 = randtool("float", -10, 10, [4, 5])
    res = np.array(
        [
            [
                [73.18701912, -11.52287175, 91.61517621, 118.53215499, 142.29236773],
                [43.21168811, -33.12990827, 7.8617612, 38.54890228, -52.8539273],
                [-81.71975865, -106.99172264, 48.05721241, 64.29614433, -21.73548278],
            ],
            [
                [11.13339829, -0.42093088, -10.65938471, -10.93244617, 3.48278416],
                [-2.21837061, -62.10974813, 41.86436987, 49.63291545, 102.91430382],
                [63.70912994, -4.02823863, 62.22807868, 79.01445048, 125.88772835],
            ],
        ]
    )
    obj.run(res=res, input=x, mat2=mat2)


@pytest.mark.api_base_mm_parameters
def test_mm4():
    """
    input: [B, M, K], mat2: [K]
    """
    x = randtool("float", -10, 10, [2, 3, 4])
    mat2 = randtool("float", -10, 10, [4])
    res = np.array([[29.30898014, 36.30260066, -45.07294524], [64.89943564, -76.69028251, -31.58245613]])
    obj.run(res=res, input=x, mat2=mat2)


@pytest.mark.api_base_mm_parameters
def test_mm5():
    """
    input: [K], mat2: [K]
    """
    x = randtool("float", -10, 10, [7])
    mat2 = randtool("float", -10, 10, [7])
    res = np.array([-102.47202965])
    obj.run(res=res, input=x, mat2=mat2)
