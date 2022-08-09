#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test lr PolynomialDecay case
"""
import paddle
import pytest


learning_rate = 0.5
decay_steps = 5
end_lr = 0.0001
power = 1.0


@pytest.mark.api_optimizer_polynomial_vartype
def test_polynomial_1():
    """
    test polynomial base test, cycle=False
    """
    scheduler_1 = paddle.optimizer.lr.PolynomialDecay(learning_rate, decay_steps, end_lr=0.0001, power=1.0, cycle=False)
    exp_list = [
        0.5,
        0.40002000000000004,
        0.30004,
        0.20006000000000002,
        0.10007999999999999,
        0.0001,
        0.0001,
        0.0001,
        0.0001,
        0.0001,
        0.0001,
        0.0001,
        0.0001,
        0.0001,
        0.0001,
        0.0001,
        0.0001,
        0.0001,
        0.0001,
        0.0001,
    ]
    for i in range(0, 20):
        exp = exp_list[i]
        assert exp == scheduler_1.get_lr()
        scheduler_1.step()


@pytest.mark.api_optimizer_polynomial_parameters
def test_polynomial_2():
    """
    test polynomial test, cycle=Treu
    """
    scheduler_1 = paddle.optimizer.lr.PolynomialDecay(learning_rate, decay_steps, end_lr=0.0001, power=1.0, cycle=True)
    exp_list = [
        0.5,
        0.40002000000000004,
        0.30004,
        0.20006000000000002,
        0.10007999999999999,
        0.0001,
        0.20006000000000002,
        0.15007,
        0.10007999999999999,
        0.050089999999999996,
        0.0001,
        0.13340666666666667,
        0.10007999999999999,
        0.06675333333333332,
        0.03342666666666666,
        0.0001,
        0.10007999999999999,
        0.07508500000000001,
        0.050089999999999996,
        0.025095000000000024,
    ]
    for i in range(0, 20):
        exp = exp_list[i]
        assert exp == scheduler_1.get_lr()
        scheduler_1.step()
