#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
engine 查表
"""

from engine.torch_eval import TorchLayerEval

# from interpreter.testing_trans import TrainTrans, EvalTrans


torch_engine_map = {
    "dy_eval": TorchLayerEval,
}
