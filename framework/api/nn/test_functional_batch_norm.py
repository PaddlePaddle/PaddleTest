#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_batchnorm
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestBatchNorm(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        # backward has bug, fix rd to fix, only forward.
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestBatchNorm(paddle.nn.functional.batch_norm)


@pytest.mark.api_nn_batch_norm_vartype
def test_batch_norm_base():
    """
    input_shape=(2,1,2,3)
    """
    x_data = np.array(
        [
            [[[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]]],
            [[[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]]],
        ]
    )

    mean = np.array([0.43857226])
    var = np.array([0.0596779])
    weight = np.array([0.39804426])
    bias = np.array([0.7379954])

    res = [
        [[[1.1581744, 0.48964378, 0.39304885], [0.92168134, 1.1956469, 0.7127977]]],
        [[[1.6213627, 1.1392108, 0.80700994], [0.66230893, 0.5825741, 1.2112563]]],
    ]

    obj.base(res=res, x=x_data, running_mean=mean, running_var=var, weight=weight, bias=bias)


@pytest.mark.api_nn_batch_norm_parameters
def test_batch_norm1():
    """
    input_shape=(2,1,2,3)
    """
    x_data = np.array(
        [
            [[[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]]],
            [[[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]]],
        ]
    )

    mean = np.array([0.43857226])
    var = np.array([0.0596779])
    weight = np.array([0.39804426])
    bias = np.array([0.7379954])

    res = [
        [[[1.1581744, 0.48964378, 0.39304885], [0.92168134, 1.1956469, 0.7127977]]],
        [[[1.6213627, 1.1392108, 0.80700994], [0.66230893, 0.5825741, 1.2112563]]],
    ]

    obj.run(res=res, x=x_data, running_mean=mean, running_var=var, weight=weight, bias=bias)


@pytest.mark.api_nn_batch_norm_parameters
def test_batch_norm2():
    """
    input_shape=(2,1,2,3), epsilon<=1e-03
    """
    x_data = np.array(
        [
            [[[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]]],
            [[[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]]],
        ]
    )

    mean = np.array([0.43857226])
    var = np.array([0.0596779])
    weight = np.array([0.39804426])
    bias = np.array([0.7379954])

    res = [
        [[[1.1581744, 0.48964378, 0.39304885], [0.92168134, 1.1956469, 0.7127977]]],
        [[[1.6213627, 1.1392108, 0.80700994], [0.66230893, 0.5825741, 1.2112563]]],
    ]

    obj.run(res=res, x=x_data, running_mean=mean, running_var=var, weight=weight, bias=bias, epsilon=1e-05)


@pytest.mark.api_nn_batch_norm_parameters
def test_batch_norm3():
    """
    input_shape=(2,1,2,3), epsilon<=1e-03, momentum=0.1
    """
    x_data = np.array(
        [
            [[[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]]],
            [[[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]]],
        ]
    )

    mean = np.array([0.43857226])
    var = np.array([0.0596779])
    weight = np.array([0.39804426])
    bias = np.array([0.7379954])

    res = [
        [[[1.1581744, 0.48964378, 0.39304885], [0.92168134, 1.1956469, 0.7127977]]],
        [[[1.6213627, 1.1392108, 0.80700994], [0.66230893, 0.5825741, 1.2112563]]],
    ]

    obj.run(
        res=res, x=x_data, running_mean=mean, running_var=var, weight=weight, bias=bias, epsilon=1e-05, momentum=0.1
    )


@pytest.mark.api_nn_batch_norm_parameters
def test_batch_norm4():
    """
    input_shape=(2,1,2,3), epsilon<=1e-03, momentum=0.9
    """
    x_data = np.array(
        [
            [[[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]]],
            [[[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]]],
        ]
    )

    mean = np.array([0.43857226])
    var = np.array([0.0596779])
    weight = np.array([0.39804426])
    bias = np.array([0.7379954])

    res = [
        [[[1.1581744, 0.48964378, 0.39304885], [0.92168134, 1.1956469, 0.7127977]]],
        [[[1.6213627, 1.1392108, 0.80700994], [0.66230893, 0.5825741, 1.2112563]]],
    ]

    obj.run(
        res=res, x=x_data, running_mean=mean, running_var=var, weight=weight, bias=bias, epsilon=1e-05, momentum=0.9
    )


@pytest.mark.api_nn_batch_norm_parameters
def test_batch_norm5():
    """
    input_shape=(2,1,2,3), epsilon<=1e-03, momentum=0.1, data_format='NCHW'
    """
    x_data = np.array(
        [
            [[[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]]],
            [[[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]]],
        ]
    )

    mean = np.array([0.43857226])
    var = np.array([0.0596779])
    weight = np.array([0.39804426])
    bias = np.array([0.7379954])

    res = [
        [[[1.1581744, 0.48964378, 0.39304885], [0.92168134, 1.1956469, 0.7127977]]],
        [[[1.6213627, 1.1392108, 0.80700994], [0.66230893, 0.5825741, 1.2112563]]],
    ]

    obj.run(
        res=res,
        x=x_data,
        running_mean=mean,
        running_var=var,
        weight=weight,
        bias=bias,
        epsilon=1e-05,
        momentum=0.1,
        data_format="NCHW",
    )


@pytest.mark.api_nn_batch_norm_parameters
def test_batch_norm6():
    """
    input_shape=(2,1,3)
    """
    x_data = np.array([[[0.6964692, 0.28613934, 0.22685145]], [[0.5513148, 0.71946895, 0.42310646]]])

    mean = np.array([0.9807642])
    var = np.array([0.6848297])
    weight = np.array([0.4809319])
    bias = np.array([0.39211753])

    res = [[[0.22689915, -0.01156456, -0.04601974]], [[0.14254248, 0.24026547, 0.06803408]]]

    obj.run(res=res, x=x_data, running_mean=mean, running_var=var, weight=weight, bias=bias)


@pytest.mark.api_nn_batch_norm_parameters
def test_batch_norm7():
    """
    input_shape=(2,1,3), epsilon<=1e-03
    """
    x_data = np.array([[[0.6964692, 0.28613934, 0.22685145]], [[0.5513148, 0.71946895, 0.42310646]]])

    mean = np.array([0.9807642])
    var = np.array([0.6848297])
    weight = np.array([0.4809319])
    bias = np.array([0.39211753])

    res = [[[0.22689915, -0.01156456, -0.04601974]], [[0.14254248, 0.24026547, 0.06803408]]]

    obj.run(res=res, x=x_data, running_mean=mean, running_var=var, weight=weight, bias=bias, epsilon=1e-05)


@pytest.mark.api_nn_batch_norm_parameters
def test_batch_norm8():
    """
    input_shape=(2,1,3), epsilon<=1e-03, data_format='NCL'
    """
    x_data = np.array([[[0.6964692, 0.28613934, 0.22685145]], [[0.5513148, 0.71946895, 0.42310646]]])

    mean = np.array([0.9807642])
    var = np.array([0.6848297])
    weight = np.array([0.4809319])
    bias = np.array([0.39211753])

    res = [[[0.22689915, -0.01156456, -0.04601974]], [[0.14254248, 0.24026547, 0.06803408]]]

    obj.run(
        res=res,
        x=x_data,
        running_mean=mean,
        running_var=var,
        weight=weight,
        bias=bias,
        epsilon=1e-05,
        data_format="NCL",
    )


@pytest.mark.api_nn_batch_norm_parameters
def test_batch_norm9():
    """
    input_shape=(2,1,3), epsilon<=1e-03
    """
    x_data = np.array([[[0.6964692, 0.28613934, 0.22685145]], [[0.5513148, 0.71946895, 0.42310646]]])

    mean = np.array([0.9807642])
    var = np.array([0.6848297])
    weight = np.array([0.4809319])
    bias = np.array([0.39211753])

    res = [[[0.22689915, -0.01156456, -0.04601974]], [[0.14254248, 0.24026547, 0.06803408]]]

    obj.run(res=res, x=x_data, running_mean=mean, running_var=var, weight=weight, bias=bias, epsilon=1e-05)
