#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test lr NaturalExpDecay case
"""
import math
import paddle
from lrbase import Runner


def naive_natural_exp_decay(lr_last, lr_0, epoch, gamma, **kwargs):
    """
    naive_natural_exp_decay
    """
    lr_last = lr_0 * math.exp(-gamma * epoch)
    return lr_last


def test_natural_exp_decay_1():
    """
    test NaturalExpDecay base test
    """
    paddle_lr = paddle.optimizer.lr.NaturalExpDecay
    runner = Runner(paddle_lr=paddle_lr, naive_func=naive_natural_exp_decay)
    runner.add_kwargs_to_dict("params_group1", gamma=0.9, last_epoch=-1, verbose=False)
    runner.run()
