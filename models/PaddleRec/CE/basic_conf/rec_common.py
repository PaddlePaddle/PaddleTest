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
Date:    2021/02/17 14:33:27
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

# PaddleRec
REPO_PaddleRec = "https://github.com/PaddlePaddle/PaddleRec.git"
BASE_BRANCH = "master"
# 16/17
rank_dnn_BRANCH = BASE_BRANCH
rank_lr_BRANCH = BASE_BRANCH
rank_gatednn_BRANCH = BASE_BRANCH
rank_fm_BRANCH = BASE_BRANCH
rank_deepfm_BRANCH = BASE_BRANCH
rank_xdeepfm_BRANCH = BASE_BRANCH
rank_wide_deep_BRANCH = BASE_BRANCH
rank_naml_BRANCH = BASE_BRANCH
rank_ffm_BRANCH = BASE_BRANCH
rank_dmr_BRANCH = BASE_BRANCH
rank_bst_BRANCH = BASE_BRANCH

rec_dy_dnn_BRANCH = BASE_BRANCH

match_dssm_BRANCH = BASE_BRANCH
match_pyramid_BRANCH = BASE_BRANCH
match_simnet_BRANCH = BASE_BRANCH

content_tagspace_BRANCH = BASE_BRANCH
content_textcnn_BRANCH = BASE_BRANCH

multitask_esmm_BRANCH = BASE_BRANCH
multitask_mmoe_BRANCH = BASE_BRANCH
multitask_ple_BRANCH = BASE_BRANCH
multitask_sharebottom_BRANCH = BASE_BRANCH

recall_ncf_BRANCH = BASE_BRANCH
recall_word2vec_BRANCH = BASE_BRANCH
recall_mind_BRANCH = BASE_BRANCH

# linux gpu下 P0的任务要跑的标签 daily  PaddleRec_Py37_Linux_Cuda10.2_FuncTest_P0_D
# 当daily为True，任务只跑日级别功能验证；当daily为False，任务切换为月级别收敛性任务
daily = True
if daily:
    EXEC_PRIORITY = ["p0", "p1"]
    EXEC_CASES = ["DATA_PROC", "TRAIN", "INFER"]
    EXEC_TAG = [
        "linux_st_gpu1",
        # "linux_st_gpu2",
        # "linux_st_cpu",
        # "linux_dy_gpu1",
        # "linux_dy_gpu2",
        # "linux_dy_cpu",
        "linux_down_data",
    ]
    EXEC_TAG_LINUX_GPU2 = [
        # "linux_st_gpu2",
        "linux_dy_gpu2",
        "linux_down_data",
    ]
    EXEC_TAG_LINUX_ALL = [
        "linux_st_gpu1",
        # "linux_st_gpu2",
        "linux_st_cpu",
        "linux_dy_gpu1",
        # "linux_dy_gpu2",
        "linux_dy_cpu",
        "linux_down_data",
    ]
else:
    # rec P0级别收敛性任务
    EXEC_PRIORITY = ["p0"]
    EXEC_CASES = ["DATA_PROC", "TRAIN", "INFER"]
    EXEC_TAG = ["linux_dy_gpu1_con", "linux_st_gpu1_con", "linux_down_data"]
    EXEC_TAG_LINUX_GPU2 = ["linux_dy_gpu1_con", "linux_st_gpu1_con", "linux_down_data"]
    EXEC_TAG_LINUX_ALL = ["linux_dy_gpu1_con", "linux_st_gpu1_con", "linux_down_data"]
