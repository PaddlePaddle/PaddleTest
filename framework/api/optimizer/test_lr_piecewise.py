#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test lr PiecewiseDecay case
"""
import paddle
import pytest

boundaries = [5, 10, 15]
values = [1.0, 0.5, 0.2, 0.33]


def naive_lr(epoch):
    """
    naive lr
    """
    if epoch < boundaries[0]:
        return values[0]
    elif epoch < boundaries[1]:
        return values[1]
    elif epoch < boundaries[2]:
        return values[2]
    else:
        return values[3]


@pytest.mark.api_optimizer_piecewise_vartype
def test_piecewise_1():
    """
    test piecewise base test
    """
    scheduler_1 = paddle.optimizer.lr.PiecewiseDecay(boundaries=boundaries, values=values, last_epoch=-1, verbose=False)
    for i in range(0, 20):
        exp = naive_lr(i)
        assert exp == scheduler_1.get_lr()
        scheduler_1.step()
