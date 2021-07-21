#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test randperm
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


# class TestRandperm(APIBase):
#     """
#     test
#     """
#     def hook(self):
#         """
#         implement
#         """
#         self.types = [np.int32, np.int64, np.float32, np.float64]
#         # self.debug = True
#         # self.static = True
#         # enable check grad
#         self.enable_backward = False
#
#
# obj = TestRandperm(paddle.randperm)
#
#
# @pytest.mark.p0
# def test_randperm_base():
#     """
#     base
#     """
#     res = np.array([0, 1, 5, 2, 4, 3])
#     n = 6
#     obj.base(res=res, n=n)
#
#
# def test_randperm():
#     """
#     default
#     """
#     res = np.array([0, 1, 6, 2, 9, 3, 5, 7, 4, 8])
#     n = 10
#     obj.run(res=res, n=n)
#
#
# def test_randperm1():
#     """
#     seed = 1
#     """
#     obj.seed = 1
#     res = np.array([4, 0, 5, 2, 7, 9, 6, 1, 3, 8])
#     n = 10
#     obj.run(res=res, n=n)
#
#
# def test_randperm2():
#     """
#     dtype = np.float32
#     """
#     obj.seed = 33
#     res = np.array([0., 1., 6., 2., 9., 3., 5., 7., 4., 8.])
#     n = 10
#     obj.run(res=res, n=n, dtype=np.float32)
#
#
# def test_randperm3():
#     """
#     exception n < 0 BUG
#     """
#     obj.seed = 33
#     res = np.array([0., 1., 6., 2., 9., 3., 5., 7., 4., 8.])
#     n = -1
#     obj.exception(etype="InvalidArgumentError", n=n, dtype=np.float32)
#
#
# def test_randperm4():
#     """
#     exception dtype = np.int8 BUG
#     """
#     obj.seed = 33
#     res = np.array([0., 1., 6., 2., 9., 3., 5., 7., 4., 8.])
#     n = -1
#     obj.exception(etype="NotFoundError", n=n, dtype=np.int8)