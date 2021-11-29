#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test lr LambdaDecay case
"""
import paddle
import pytest


learning_rate = 0.5


@pytest.mark.api_optimizer_warmup_vartype
def test_warmup_1():
    """
    test warmup base test
    """
    scheduler_1 = paddle.optimizer.lr.LambdaDecay(
        learning_rate=learning_rate, lr_lambda=lambda x: 0.95 ** x, last_epoch=-1, verbose=False
    )
    for i in range(0, 20):
        exp = learning_rate * 0.95 ** i
        assert exp == scheduler_1.get_lr()
        scheduler_1.step()
