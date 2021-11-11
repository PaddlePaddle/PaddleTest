#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test lr NoamDecay case
"""
import paddle
import pytest
from lrbase import Runner


def naive_noam_decay(lr_last, lr_0, epoch, d_model=0.01, warmup_steps=100, **kwargs):
    """
    naive_noam_decay
    """
    lr_last = lr_0 * (d_model ** -0.5) * min(epoch ** -0.5, epoch * warmup_steps ** -1.5)
    return lr_last


@pytest.mark.api_optimizer_noam_decay_vartype
def test_noam_decay_1():
    """
    test NoamDecay base test
    """
    paddle_lr = paddle.optimizer.lr.NoamDecay
    runner = Runner(paddle_lr=paddle_lr, naive_func=naive_noam_decay)
    runner.add_kwargs_to_dict("params_group1", d_model=0.01, warmup_steps=100, last_epoch=-1, verbose=False)
    runner.run()
