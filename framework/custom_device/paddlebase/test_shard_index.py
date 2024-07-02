#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_shard_index
"""

import random
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestShardIndex(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False
        # self.delta = 1


obj = TestShardIndex(paddle.shard_index)


def cal_shard_index(input, index_num, nshards, shard_id, ignore_value=-1):
    """
    calculate shard_index api
    """
    shard_size = (index_num + nshards - 1) // nshards
    x = input.flatten()
    output = []
    for item in x:
        if shard_id == item // shard_size:
            output.append(item % shard_size)
        else:
            output.append(ignore_value)
    return np.array(output).reshape(input.shape)


@pytest.mark.api_base_shard_index_vartype
def test_shard_index_base():
    """
    base
    """
    x = np.random.randint(4, 13, (4, 1))
    res = cal_shard_index(x, 3, 13, 0)
    obj.base(res=res, input=x, index_num=13, nshards=3, shard_id=0)


@pytest.mark.api_base_shard_index_parameters
def test_shard_index0():
    """
    default
    """
    x = np.random.randint(0, 7, (2, 1))
    res = cal_shard_index(x, 20, 2, 1)
    obj.run(res=res, input=x, index_num=20, nshards=2, shard_id=1)


@pytest.mark.api_base_shard_index_parameters
def test_shard_index1():
    """
    input: multiple dimension
    """
    x = np.random.randint(2, 17, (4, 2, 1))
    res = cal_shard_index(x, 20, 4, 1)
    obj.run(res=res, input=x, index_num=20, nshards=4, shard_id=1)


@pytest.mark.api_base_shard_index_parameters
def test_shard_index2():
    """
    input: multiple dimension
    ignore_value: random
    """
    x = np.random.randint(2, 17, (4, 2, 1))
    ignore_value = random.randint(-10, 20)
    res = cal_shard_index(x, 20, 4, 1, ignore_value)
    obj.run(res=res, input=x, index_num=20, nshards=4, shard_id=1, ignore_value=ignore_value)


@pytest.mark.api_base_shard_index_parameters
def test_shard_index3():
    """
    nshards >> index_num
    """
    x = np.random.randint(2, 5, (2, 1))
    res = cal_shard_index(x, 6, 40, 4)
    obj.run(res=res, input=x, index_num=6, nshards=40, shard_id=4)
