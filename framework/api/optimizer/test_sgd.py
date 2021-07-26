#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test sgd case
"""


import paddle
from runner import Runner
from base_dygraph_model import Dygraph
import reader


def test_sgd():
    """
    test sgd base test
    """
    model = Dygraph()
    datareader = reader.reader
    optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    runner = Runner(datareader, model, optimizer)
    # runner.debug = True
    runner.run()
    # expect = [
    #     2.6125570230480717,
    #     2.6106062484608996,
    #     2.606704699286556,
    #     2.6008523755250392,
    #     2.593049277176351,
    #     2.5832954042404905,
    #     2.5715907567174576,
    #     2.5579353346072526,
    #     2.542329137909876,
    #     2.524772166625327,
    # ]
    expect = [
        2.6125570230480717,
        2.6106062484608996,
        2.608655473873728,
        2.6067046992865555,
        2.604753924699384,
        2.6028031501122113,
        2.6008523755250392,
        2.598901600937867,
        2.596950826350695,
        2.595000051763523,
    ]
    runner.check(expect)
