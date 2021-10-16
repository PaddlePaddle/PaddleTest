# encoding: utf-8
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
mac cpu daily
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
# PaddleHub
REPO_PaddleHub = "https://github.com/PaddlePaddle/PaddleHub.git"
BASE_BRANCH = "develop"
NLP_BRANCH = BASE_BRANCH
CV_BRANCH = BASE_BRANCH
AUDIO_BRANCH = BASE_BRANCH

EXEC_PRIORITY = ["p0", "p1"]
EXEC_CASES = ["INSTALL", "FINETUNE", "PREDICT", "UNINSTALL"]
EXEC_TAG = [
    # 补充一些自定义标签
    "install_module",
    "mac_predict_without_model_cpu",
    "uninstall_module",
]
