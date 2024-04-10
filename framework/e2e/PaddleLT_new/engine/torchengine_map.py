#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
engine 查表
"""

from engine.train import LayerTrain
from engine.eval import LayerEval
from engine.export import LayerExport
from engine.infer import LayerInfer
from engine.eval_bm import LayerEvalBM

from engine.torcheval import TorchLayerEval

# from interpreter.testing_trans import TrainTrans, EvalTrans


engine_map = {
    "torch_dy_eval": TorchLayerEval,
}
