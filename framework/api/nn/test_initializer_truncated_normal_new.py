#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
# """
# test initializer_truncated_normal
# """
# from apibase import APIBase
# from apibase import compare
# import pytest
# import paddle
# import numpy as np
#
#
# @pytest.mark.api_initializer_truncated_normal_vartype
# def test_initializer_truncated_normal_base():
#    """
#    base
#    """
#    w_init = paddle.nn.initializer.TruncatedNormal()
#    Linear = paddle.nn.Linear(10000, 10000, weight_attr=w_init)
#    w = Linear._parameters["weight"].numpy()
#
#    compare(np.array(np.std(w)), 0.879, delta=1e-2, rtol=1e-2)
#    compare(np.std(w, axis=0), np.ones(10000) * 0.879, delta=1e-1, rtol=1e-1)
#    compare(np.std(w, axis=1), np.ones(10000) * 0.879, delta=1e-1, rtol=1e-1)
#    compare(np.array(np.mean(w)), 0, delta=1e-1, rtol=1e-1)
#    compare(np.mean(w, axis=0), np.zeros(10000), delta=1e-1, rtol=1e-1)
#    compare(np.mean(w, axis=1), np.zeros(10000), delta=1e-1, rtol=1e-1)
#
#
# @pytest.mark.api_initializer_truncated_normal_parameters
# def test_initializer_truncated_normal1():
#    """
#    base
#    """
#    w_init = paddle.nn.initializer.TruncatedNormal(mean=-2.0, std=5.0)
#    Linear = paddle.nn.Linear(10000, 10000, weight_attr=w_init)
#    w = Linear._parameters["weight"].numpy()
#
#    compare(np.array(np.std(w)), 0.879 * 5, delta=1e-2, rtol=1e-2)
#    compare(np.std(w, axis=0), np.ones(10000) * 0.879 * 5, delta=1e-1, rtol=1e-1)
#    compare(np.std(w, axis=1), np.ones(10000) * 0.879 * 5, delta=1e-1, rtol=1e-1)
#    compare(np.array(np.mean(w)), -2.0, delta=1e-1, rtol=1e-1)
#    compare(np.mean(w, axis=0), np.ones(10000) * -2.0, delta=1e-1, rtol=1e-1)
#    compare(np.mean(w, axis=1), np.ones(10000) * -2.0, delta=1e-1, rtol=1e-1)
