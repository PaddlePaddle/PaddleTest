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
PROJECT_ROOT = os.path.dirname(ROOT_PATH)
# 公共全局变量
PADDLE_ON_MODEL_CE = "1"
WITH_AVX = "ON"
# 资源配置
XPU = "gpu"  # 取值gpu或cpu
# 配置文件中SET_CUDA为None，框架不配置gpu cuda，使用cpu场景
SET_CUDA = "0"
SET_MULTI_CUDA = "0,1"

# PaddleNLP
REPO_PaddleNLP = "https://github.com/PaddlePaddle/PaddleNLP.git"
BASE_BRANCH = "develop"
# windows GPU 任务要跑的标签
EXEC_PRIORITY = ["p0", "p1"]
EXEC_CASES = ["DATA_PROC", "TRAIN", "INFER"]
EXEC_TAG = [
    "win_st_gpu1",
    "win_st_gpu1_con",
    "win_dy_gpu1",
    "win_dy_gpu1_con",
    "win_download_data",
    "win_gpu1_SST-2",
    "win_gpu1_MRPC",
    "win_gpu1_STS-B",
    "win_gpu1_QNLI",
    "win_gpu1_QQP",
    "win_gpu1_MNLI",
    "win_gpu1_RTE",
    "win_gpu1_WNLI",
    "win_dy_gpu1_batch",
    "win_dy_gpu1_hardest",
    "win_gpu1_CoLA",
    "win_dy_gpu1_iflytek",
    "win_dy_gpu1_tnews",
    "win_dy_gpu1_eprstmt",
    "win_dy_gpu1_bustm",
    "win_dy_gpu1_ocnli",
    "win_dy_gpu1_csl",
    "win_dy_gpu1_csldcp",
    "win_dy_gpu1_cluewsc",
    "win_dy_gpu1_chid",
    "win_dy_gpu1_trigger",
    "win_dy_gpu1_role",
    "win_dy_gpu1_enum",
    "win_dy_gpu1_ernie",
    "win_dy_gpu1_ernie_crf",
    "win_dy_gpu1_bigru_crf",
    "win_dy_gpu1_24",
    "win_dy_gpu1_thucnews",
    "win_dy_gpu1_imdb",
    "win_dy_gpu1_hyp",
    "win_dy_gpu1_dureader_robust",
    "win_dy_gpu1_drcd",
    "win_dy_gpu1_cmrc2018",
    "win_dy_gpu1_afqmc",
    "win_dy_gpu1_pair-wise",
    "win_dy_gpu1_point-wise",
    "win_dy_gpu1_ATEC",
    "win_dy_gpu1_STS-B",
    "win_dy_gpu1_BQ_Corpus",
    "win_dy_gpu1_LCQMC",
    "win_dy_gpu1_sst-2",
    "win_dy_gpu1_CHIP-STS",
]
