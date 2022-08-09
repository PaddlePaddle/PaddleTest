#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test lr MultiplicativeDecay
"""
import paddle
import pytest
import numpy as np


def naive_multiplicative_deacy(lr, lr_lambda):
    """
    naive_MultiplicativeDecay
    """
    obj = lr_lambda
    lr = obj(lr)
    return lr


@pytest.mark.api_optimizer_MultiplicativeDecay_vartype
def test_MultiplicativeDecayy_0():
    """
    test MultiplicativeDecay
    """
    scheduler = paddle.optimizer.lr.MultiplicativeDecay(learning_rate=1, lr_lambda=lambda x: 0.1, verbose=False)
    res_lr = scheduler.get_lr()
    for i in range(20):
        exp_lr = naive_multiplicative_deacy(res_lr, lambda x: 0.1 * x)
        scheduler.step()
        res_lr = scheduler.get_lr()
        assert np.allclose(res_lr, exp_lr)


@pytest.mark.api_optimizer_MultiplicativeDecay_vartype
def test_MultiplicativeDecayy_1():
    """
    test MultiplicativeDecay 1
    """
    scheduler = paddle.optimizer.lr.MultiplicativeDecay(learning_rate=0.2, lr_lambda=lambda x: 0.95, verbose=False)
    res_lr = scheduler.get_lr()
    for i in range(20):
        exp_lr = naive_multiplicative_deacy(res_lr, lambda x: 0.95 * x)
        scheduler.step()
        res_lr = scheduler.get_lr()
        assert np.allclose(res_lr, exp_lr)


@pytest.mark.api_optimizer_MultiplicativeDecay_vartype
def test_MultiplicativeDecay_2():
    """
    verbose=True
    """
    scheduler = paddle.optimizer.lr.MultiplicativeDecay(learning_rate=1, lr_lambda=lambda x: 0.1, verbose=True)
    res_lr = scheduler.get_lr()
    for i in range(20):
        exp_lr = naive_multiplicative_deacy(res_lr, lambda x: 0.1 * x)
        scheduler.step()
        res_lr = scheduler.get_lr()
        assert np.allclose(res_lr, exp_lr)
