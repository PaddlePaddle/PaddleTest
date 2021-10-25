#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test bernoulli
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
from paddle import fluid
import numpy as np


class TestBernoulli(APIBase):
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
        self.places = [fluid.CPUPlace()]
        self.enable_backward = False


obj = TestBernoulli(paddle.bernoulli)


@pytest.mark.api_base_bernoulli_vartype
def test_bernoulli_base():
    """
    test base
    Returns:

    """
    x = randtool("float", 0, 1, shape=[6, 3])
    res = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]
    )
    obj.run(res=res, x=x)
