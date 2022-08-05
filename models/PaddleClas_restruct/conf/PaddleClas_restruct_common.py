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


EXEC_PRIORITY = ["precision"]
# EXEC_PRIORITY = ["precision", "function"]
#precision表示精度，有指标评估步骤
#function表示功能性，只关注是否报错



EXEC_TAG = ["train_linux_gpu1", "train_linux_gpu2", "eval_linux", "infer_linux", "export_linux", "predict_linux"]

# Linux 不增加预训练模型
#develop
# EXEC_TAG = ["train_linux_gpu1","train_linux_gpu2","eval_linux","infer_linux","export_linux","predict_linux"]
#release
# EXEC_TAG = ["train_linux_gpu1_release","train_linux_gpu2_release","eval_linux_release",
#   "infer_linux_release","export_linux_release","predict_linux_release"]

# Linux 加载预训练模型
#develop
# EXEC_TAG = ["train_linux_gpu1","train_linux_gpu2","eval_linux","eval_linux_pretrained","infer_linux",
#   "infer_linux_pretrained","export_linux","export_linux_pretrained","predict_linux","predict_linux_pretrained"]
#release
# EXEC_TAG = ["train_linux_gpu1_release","train_linux_gpu2_release","eval_linux_release","eval_linux_pretrained",
# "infer_linux_release","infer_linux_pretrained","export_linux","export_linux_pretrained","predict_linux_release","predict_linux_pretrained"]

# Windows 执行步骤
# EXEC_TAG = ["win_function_test", "train_win_gpu", "eval_win", "infer_win", "predict_win"]

# MAC 执行步骤
# EXEC_TAG = ["mac_function_test", "train_mac_cpu", "eval_mac", "infer_mac", "predict_mac"]

# Linux 收敛性
# EXEC_TAG = ["train_linux_convergence"]
