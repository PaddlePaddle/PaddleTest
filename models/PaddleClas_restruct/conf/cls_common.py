# encoding: utf-8
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件提供了代码中用到的公共配置。
Creators: paddlepaddle-qa
Date:    2021/04/21 14:33:27
"""

import os


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
# 公共全局变量
PADDLE_ON_MODEL_CE = "1"
WITH_AVX = "ON"
# 资源配置
IS_SINGLE_CUDA = True
XPU = "gpu"  # 取值gpu或cpu
# 配置文件中SET_CUDA为None，框架不配置gpu cuda，使用cpu场景
SET_CUDA = "0"
SET_MULTI_CUDA = "0,1"
# if XPU == 'gpu':
#     if IS_SINGLE_CUDA:
#         SET_CUDA = '0'
#     else:
#         SET_CUDA = '0'
#         SET_MULTI_CUDA = '0,1'

# PaddleClas
REPO_PaddleClas = "https://github.com/PaddlePaddle/PaddleClas.git"
BASE_BRANCH = "develop"
cls_BRANCH = "develop"
# EXEC_PRIORITY = ["precision", "function"]
EXEC_PRIORITY = ["precision"]
EXEC_TAG = ["train_linux_gpu1", "train_linux_gpu2", "eval_linux", "infer_linux", "export_linux", "predict_linux"]

# EXEC_TAG = [
#     "train_linux_gpu1",
#     "train_linux_gpu2",
#     "eval_linux",
#     "eval_linux_pretrained",
#     "infer_linux",
#     "infer_linux_pretrained",
#     "export_linux",
#     "export_linux_pretrained",
#     "predict_linux",
#     "predict_linux_pretrained",
# ]
# EXEC_TAG = ["train_linux_gpu1", "train_linux_gpu2", "eval_linux", "infer_linux", "predict_linux"]
