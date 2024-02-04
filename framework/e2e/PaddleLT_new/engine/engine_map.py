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

# from interpreter.testing_trans import TrainTrans, EvalTrans


engine_map = {
    "dy_train": LayerTrain,
    "dy2st_train": LayerTrain,
    "dy2st_train_cinn": LayerTrain,
    "dy_eval": LayerEval,
    "dy2st_eval": LayerEval,
    "dy2st_eval_cinn": LayerEval,
    "jit_save": LayerExport,
    "paddle_infer_gpu": LayerInfer,
    "paddle_infer_cpu": LayerInfer,
    "paddle_infer_mkldnn": LayerInfer,
    "paddle_infer_ort": LayerInfer,
}
