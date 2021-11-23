#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_graph_send_recv
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestGraphSendRecv(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestGraphSendRecv(paddle.incubate.graph_send_recv)


@pytest.mark.api_nn_graph_send_recv_vartype
def test_graph_send_recv_base():
    """
    base
    """
    x = np.array([[0, 2, 3], [1, 4, 5], [2, 6, 7]])
    src_index = np.array([0, 1, 2, 0])
    dst_index = np.array([1, 2, 1, 0])
    res = np.array([[0.0, 2.0, 3.0], [2.0, 8.0, 10.0], [1.0, 4.0, 5.0]])
    obj.base(res=res, x=x, src_index=src_index, dst_index=dst_index)


@pytest.mark.api_nn_graph_send_recv_parameters
def test_graph_send_recv0():
    """
    pool_type='mean'
    """
    x = np.array([[0, 2, 3], [1, 4, 5], [2, 6, 7]])
    src_index = np.array([0, 1, 2, 0])
    dst_index = np.array([1, 2, 1, 0])
    res = np.array([[0.0, 2.0, 3.0], [1, 4, 5], [1.0, 4.0, 5.0]])
    obj.run(res=res, x=x, src_index=src_index, dst_index=dst_index, pool_type="mean")


@pytest.mark.api_nn_graph_send_recv_parameters
def test_graph_send_recv1():
    """
    pool_type='max'
    """
    x = np.array([[0, 2, 3], [1, 4, 5], [2, 6, 7]])
    src_index = np.array([0, 1, 2, 0])
    dst_index = np.array([1, 2, 1, 0])
    res = np.array([[0.0, 2.0, 3.0], [2, 6, 7], [1.0, 4.0, 5.0]])
    obj.base(res=res, x=x, src_index=src_index, dst_index=dst_index, pool_type="max")


@pytest.mark.api_nn_graph_send_recv_parameters
def test_graph_send_recv2():
    """
    pool_type='min'
    """
    x = np.array([[0, 2, 3], [1, 4, 5], [2, 6, 7]])
    src_index = np.array([0, 1, 2, 0])
    dst_index = np.array([1, 2, 1, 0])
    res = np.array([[0.0, 2.0, 3.0], [0, 2, 3], [1.0, 4.0, 5.0]])
    obj.base(res=res, x=x, src_index=src_index, dst_index=dst_index, pool_type="min")
