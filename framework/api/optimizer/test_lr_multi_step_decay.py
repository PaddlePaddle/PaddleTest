#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test lr MultiStepDecay case
"""
import paddle
from lrbase import Runner


def naive_multi_step_decay(lr_last, lr_0, epoch, milestones, gamma, **kwargs):
    """
    naive_multi_step_decay
    """
    for i, v in enumerate(milestones):
        if epoch < v:
            return lr_0 * (gamma ** i)
    return lr_0 * (gamma ** len(milestones))


def test_multi_step_decay_1():
    """
    test MultiStepDecay base test
    """
    paddle_lr = paddle.optimizer.lr.MultiStepDecay
    runner = Runner(paddle_lr=paddle_lr, naive_func=naive_multi_step_decay)
    runner.add_kwargs_to_dict("params_group1", milestones=[2, 4, 10, 11], gamma=0.1, last_epoch=-1, verbose=False)
    runner.run()
