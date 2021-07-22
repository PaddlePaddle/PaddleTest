#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test round
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestRound(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.rtol = 0.1
        self.gap = 0.0001
        # self.debug = True
        # self.static = False
        # enable check grad
        # self.enable_backward = True


obj = TestRound(paddle.round)


@pytest.mark.api_base_round_vartype
def test_round_base():
    """
    base
    """
    x = -1 + 2 * np.random.random(size=[3, 3, 3])
    res = np.round(x)
    obj.base(res=res, x=x)
