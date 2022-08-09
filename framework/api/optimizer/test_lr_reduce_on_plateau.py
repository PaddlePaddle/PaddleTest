#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test lr ReduceOnPlateau case
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
import reader


@pytest.mark.api_optimizer_reduce_on_plateau_vartype
def test_reduce_on_plateau_1():
    """
    learning_rate=0.5, mode='min', factor=0.1,
    patience=3, threshold=1e-4, threshold_mode='rel',
    cooldown=0, min_lr=0, epsilon=1e-8
    """
    model = Dygraph()
    datareader = reader.reader
    scheduler = paddle.optimizer.lr.ReduceOnPlateau(
        learning_rate=0.5,
        mode="min",
        factor=0.1,
        patience=3,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        epsilon=1e-8,
    )
    optimizer = paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters())
    runner = Runner(datareader, model, optimizer)
    # runner.debug = True
    runner.run()
    expect = [
        2.6125570230480717,
        1.6371697294620158,
        0.6617824358759596,
        -0.31360485771009655,
        -1.2889921512961526,
        -2.2643794448822088,
        -3.2397667384682647,
        -4.215154032054321,
        -5.190541325640377,
        -6.165928619226433,
    ]
    runner.check(expect)


@pytest.mark.api_optimizer_reduce_on_plateau_parameters
def test_reduce_on_plateau_2():
    """
    learning_rate=0.5, mode='max',
    factor=0.1, patience=5,
    threshold=1e-4, threshold_mode='rel',
    cooldown=0, min_lr=0, epsilon=1e-8
    """
    model = Dygraph()
    datareader = reader.reader
    scheduler = paddle.optimizer.lr.ReduceOnPlateau(
        learning_rate=0.5,
        mode="max",
        factor=0.1,
        patience=5,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        epsilon=1e-8,
    )
    optimizer = paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters())
    runner = Runner(datareader, model, optimizer)
    # runner.debug = True
    runner.run()
    expect = [
        2.6125570230480717,
        1.6371697294620158,
        0.6617824358759596,
        -0.31360485771009655,
        -1.2889921512961526,
        -2.2643794448822088,
        -3.2397667384682647,
        -4.215154032054321,
        -5.190541325640377,
        -6.165928619226433,
    ]
    runner.check(expect)
