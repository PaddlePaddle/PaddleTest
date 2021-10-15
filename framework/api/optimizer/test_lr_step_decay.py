#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test lr StepDecay case
"""
import paddle
import pytest
from lrbase import Runner


def naive_step_decay(lr_last, lr_0, epoch, step_size, gamma, **kwargs):
    """
    naive_step_decay
    """
    i = epoch // step_size
    return lr_0 * (gamma ** i)


@pytest.mark.api_optimizer_step_decay_vartype
def test_step_decay_1():
    """
    test StepDecay base test
    """
    paddle_lr = paddle.optimizer.lr.StepDecay
    runner = Runner(paddle_lr=paddle_lr, naive_func=naive_step_decay)
    runner.add_kwargs_to_dict("params_group1", step_size=3, gamma=0.1, last_epoch=-1, verbose=False)
    runner.run()
