#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test AdamW case
"""
import paddle
import pytest
from runner import Runner
from base_dygraph_model import Dygraph
from linear_dygraph_model import LinearNet
import reader


@pytest.mark.api_optimizer_adamw_vartype
def test_adamw_base():
    """
    test adamw base test
    """
    model = Dygraph()
    datareader = reader.reader
    optimizer = paddle.optimizer.AdamW(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        parameters=model.parameters(),
        weight_decay=0.01,
        grad_clip=None,
        name=None,
        lazy_mode=False,
    )
    runner = Runner(datareader, model, optimizer)
    runner.run()
    expect = [
        2.6125570230480717,
        2.6069657465008333,
        2.601374544485957,
        2.5957834045887838,
        2.59019232370521,
        2.5846013005932567,
        2.5790103346316533,
        2.5734194254651532,
        2.5678285728715196,
        2.562237776702413,
    ]
    runner.check(expect)
