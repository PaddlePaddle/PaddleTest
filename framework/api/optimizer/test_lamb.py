#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test Lamb case
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
from linear_dygraph_model import LinearNet
import reader


@pytest.mark.api_optimizer_lamb_vartype
def test_lamb_base():
    """
    test lamb base test
    """
    model = Dygraph()
    datareader = reader.reader
    optimizer = paddle.optimizer.Lamb(
        learning_rate=0.001,
        lamb_weight_decay=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-06,
        parameters=model.parameters(),
        grad_clip=None,
        exclude_from_weight_decay_fn=None,
    )
    runner = Runner(datareader, model, optimizer)
    runner.run()
    expect = [
        2.6125570230480717,
        2.6100731760313085,
        2.607439563538132,
        2.6048123726137815,
        2.602192184882482,
        2.599577698693398,
        2.5969679539573978,
        2.5943623477791036,
        2.591760493591688,
        2.5891621312693403,
    ]
    runner.check(expect)
