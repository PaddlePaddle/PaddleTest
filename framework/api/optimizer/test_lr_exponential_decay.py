#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test lr ExponentialDecay case
"""
import paddle
from lrbase import Runner


def naive_exponential_decay(lr_last, lr_0, epoch, gamma, **kwargs):
    """
    naive_exponential_decay
    """
    lr_last = lr_last * gamma
    return lr_last


def test_exponential_decay_1():
    """
    test ExponentialDecay base test
    """
    paddle_lr = paddle.optimizer.lr.ExponentialDecay
    runner = Runner(paddle_lr=paddle_lr, naive_func=naive_exponential_decay)
    runner.add_kwargs_to_dict("params_group1", gamma=0.9, last_epoch=-1, verbose=False)
    runner.run()


# def test_ExponentialDecay_2():
#     """
#     test step > ExponentialDecay_step
#     """
#     last_epoch = ExponentialDecay_steps + 1
#     scheduler_2 = paddle.optimizer.lr.LinearWarmup(
#         learning_rate=learning_rate,
#         ExponentialDecay_steps=ExponentialDecay_steps,
#         start_lr=start_lr,
#         end_lr=end_lr,
#         last_epoch=last_epoch,
#         verbose=False,
#     )
#     assert learning_rate == scheduler_2.get_lr()
#
#
# def test_ExponentialDecay_3():
#     """
#     test ExponentialDecay_steps <=0
#     """
#     try:
#         paddle.optimizer.lr.LinearWarmup(
#             learning_rate=learning_rate, ExponentialDecay_steps=0, start_lr=start_lr, end_lr=end_lr, verbose=True
#         )
#     except AssertionError as error:
#         assert """'ExponentialDecay_steps' must be a positive integer""" in str(error)
