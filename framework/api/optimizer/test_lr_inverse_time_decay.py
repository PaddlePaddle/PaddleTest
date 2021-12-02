#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test lr InverseTimeDecay case
"""
import paddle
import pytest
from lrbase import Runner


def naive_inverse_time_decay(lr_last, lr_0, epoch, gamma, **kwargs):
    """
    naive_inverse_time_decay
    """
    lr_last = lr_0 / (1 + gamma * epoch)
    return lr_last


@pytest.mark.api_optimizer_inverse_time_decay_vartype
def test_inverse_time_decay_1():
    """
    test InverseTimeDecay base test
    """
    paddle_lr = paddle.optimizer.lr.InverseTimeDecay
    runner = Runner(paddle_lr=paddle_lr, naive_func=naive_inverse_time_decay)
    runner.add_kwargs_to_dict("params_group1", gamma=0.9, last_epoch=-1, verbose=False)
    runner.run()
