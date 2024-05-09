#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_eigvals
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestEigvals(APIBase):
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
        self.enable_backward = False


obj = TestEigvals(paddle.linalg.eigvals)
obj.places = [paddle.CPUPlace()]


@pytest.mark.api_linalg_eigvals_vartype
def test_eigvals_base():
    """
    base: float32 different with float64
    """
    obj.delta = 1e-4
    x = np.array([[2, 4, 5, 1], [2, 3, 4, 0], [4, 1, 2, 3], [5, 0, 2, 5]], dtype=np.float32)
    res = np.array(
        [(10.820149421691895 + 0j), (2.965639591217041 + 0j), (-1.410071611404419 + 0j), (-0.3757128417491913 + 0j)]
    )
    obj.base(res=res, x=x)


@pytest.mark.api_linalg_eigvals_parameters
def test_eigvals0():
    """
    default
    """

    x = np.array(
        [
            [19.15194504, 62.21087710, 43.77277390],
            [78.53585837, 77.99758081, 27.25926053],
            [27.64642551, 80.18721775, 95.81393537],
        ]
    )
    res = np.array([(173.88151005956013 + 0j), (-20.270675701439156 + 0j), (39.35262686001911 + 0j)])
    obj.run(res=res, x=x)


@pytest.mark.api_linalg_eigvals_parameters
def test_eigvals1():
    """
    multi_dim
    """
    x = np.array(
        [
            [
                [1.91519450, 6.22108771, 4.37727739, 7.85358584],
                [7.79975808, 2.72592605, 2.76464255, 8.01872178],
                [9.58139354, 8.75932635, 3.57817270, 5.00995126],
                [6.83462935, 7.12702027, 3.70250755, 5.61196186],
            ],
            [
                [5.03083165, 0.13768450, 7.72826622, 8.82641191],
                [3.64885984, 6.15396178, 0.75381242, 3.68824006],
                [9.33140102, 6.51378143, 3.97202578, 7.88730143],
                [3.16836122, 5.68098653, 8.69127390, 4.36173424],
            ],
        ]
    )
    res = np.array(
        [
            [
                (22.543468523447373 + 0j),
                (-4.241885468199413 + 0.6086924519661563j),
                (-4.241885468199413 - 0.6086924519661563j),
                (-0.22844247019828132 + 0j),
            ],
            [
                (21.70537198445617 + 0j),
                (3.662687194877007 + 0j),
                (-2.9247528628509065 + 2.614455712266121j),
                (-2.9247528628509065 - 2.614455712266121j),
            ],
        ]
    )
    obj.run(res=res, x=x)
