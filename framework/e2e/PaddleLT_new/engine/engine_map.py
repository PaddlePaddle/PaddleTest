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
    "dy_train": LayerTrain,  # 动态图训练
    "dy2st_train": LayerTrain,  # 动转静训练
    "dy2st_train_cinn": LayerTrain,  # CINN训练
    "dy_eval": LayerEval,  # 动态图评估
    "dy2st_eval": LayerEval,  # 动转静评估
    "dy2st_eval_cinn": LayerEval,  # CINN评估
    "jit_save": LayerExport,  # 动转静导出
    "paddle_infer_gpu": LayerInfer,  # gpu预测
    "paddle_infer_cpu": LayerInfer,  # cpu预测
    "paddle_infer_mkldnn": LayerInfer,  # cpu mkldnn预测
    "paddle_infer_ort": LayerInfer,  # ort预测
    "dy_eval_perf": LayerEvalBM,  # 动态图评估性能
    "dy2st_eval_cinn_perf": LayerEvalBM,  # CINN评估性能

    "torch_dy_eval": TorchLayerEval,
}
