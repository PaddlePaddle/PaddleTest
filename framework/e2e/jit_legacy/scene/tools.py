#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
tools
"""
import os
import logging
import numpy as np
import pytest


def compare(result, expect, delta=1e-5, rtol=1e-4):
    """
    比较函数
    :param result: 输入值
    :param expect: 输出值
    :param delta: 误差值
    :return:
    """
    if isinstance(result, np.ndarray):
        expect = np.array(expect)
        res = np.allclose(result, expect, atol=delta, rtol=rtol, equal_nan=True)
        # 出错打印错误数据
        if res is False:
            logging.error("the result is {}".format(result))
            logging.error("the expect is {}".format(expect))
        # tools.assert_true(res)
        assert res
        # tools.assert_equal(result.shape, expect.shape)
        assert result.shape == expect.shape
    elif isinstance(result, list):
        result = np.array(result)
        expect = np.array(expect)
        res = np.allclose(result, expect, atol=delta)
        # 出错打印错误数据
        if res is False:
            print("the result is {}".format(result))
            print("the expect is {}".format(expect))
        # tools.assert_true(res)
        assert res
        # tools.assert_equal(result.shape, expect.shape)
        assert result.shape == expect.shape
    elif isinstance(result, str):
        res = result == expect
        if res is False:
            logging.error("the result is {}".format(result))
            logging.error("the expect is {}".format(expect))
        assert res
    else:
        assert result == pytest.approx(expect, delta)
        # tools.assert_almost_equal(result, expect, delta=delta)


def randtool(dtype, low, high, shape):
    """
    np random tools
    """
    if dtype == "int":
        return np.random.randint(low, high, shape)

    elif dtype == "float":
        return low + (high - low) * np.random.random(shape)


def delete_all(path):
    """
    delete all files under path
    """
    for i in os.listdir(path):
        if i == "readme.md" or i == "input_np.npy":
            continue
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            delete_all(path_file)
