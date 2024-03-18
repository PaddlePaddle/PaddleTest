#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test index_sample
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestIndexSample(APIBase):
    """
    test nonzero
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64, np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.no_grad_var = ["index"]
        self.enable_backward = False


obj = TestIndexSample(paddle.index_sample)


@pytest.mark.api_base_index_sample_vartype
def test_index_sample():
    """
    index_sample base
    """
    x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
    index = np.array([[0, 1, 2], [1, 2, 3], [0, 0, 0]])
    res = np.array([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0], [9.0, 9.0, 9.0]])
    obj.base(res=res, x=x, index=index)


# 易用性问题
# @pytest.mark.api_base_index_sample_vartype
# def test_index_sample2():
#     """
#     index_sample base
#     """
#     for t in [np.float32, np.float64]:
#         x = np.array([[],
#                       [],
#                       []]).astype(t)
#         index = np.array([0, 0, 0])
#         res = np.array([[],
#                         [],
#                         []])
#         obj.run(res=res, x=x, index=index)
