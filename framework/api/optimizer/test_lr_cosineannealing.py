#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test lr CosineAnnealingDecay case
"""
import paddle
import pytest


@pytest.mark.api_optimizer_cosineannealing_vartype
def test_cosineannealing_1():
    """
    test cosineannealing base test, T_max=10, eta_min=0
    """
    scheduler_1 = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.5, T_max=10, eta_min=0, last_epoch=-1)
    exp_list = [
        0.5,
        0.4758276912222226,
        0.4193295348706216,
        0.3484022006477268,
        0.26979805504056703,
        0.19098300562505258,
        0.11936437851565788,
        0.06147799469896954,
        0.022121059860440846,
        0.0031357038671145786,
        0.0,
        0.024471741852423234,
        0.18630931881319765,
        0.2224294743810586,
        0.2895684329090268,
        0.36180339887499097,
        0.4283813728906071,
        0.48147999953989984,
        0.515268434634411,
        0.5260621571845796,
    ]

    for i in range(0, 20):
        exp = exp_list[i]
        assert exp == scheduler_1.get_lr()
        scheduler_1.step()


@pytest.mark.api_optimizer_cosineannealing_parameters
def test_cosineannealing_2():
    """
    test cosineannealing test, T_max=5, eta_min=0.001
    """
    scheduler_1 = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.5, T_max=5, eta_min=0.001, last_epoch=-1)
    exp_list = [
        0.5,
        0.4092496751206867,
        0.23732979207723948,
        0.09200389855175925,
        0.0141702079227605,
        0.001,
        0.09630051980690123,
        0.6247500000000005,
        0.6197194282124089,
        0.6247500000000004,
        0.5526808316910422,
        0.4092496751206872,
        0.2373297920772396,
        0.09200389855175936,
        0.014170207922760518,
        0.001,
        0.09630051980690123,
        0.6247500000000007,
        0.6197194282124094,
        0.6247500000000008,
    ]
    for i in range(0, 20):
        exp = exp_list[i]
        assert exp == scheduler_1.get_lr()
        scheduler_1.step()
