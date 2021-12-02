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
# daily P0任务要跑的标签
EXEC_PRIORITY = ["p0", "p1"]
EXEC_CASES = ["DATA_PROC", "TRAIN", "INFER"]
EXEC_TAG = [
    "linux_st_gpu1",
    "linux_st_gpu2",
    "linux_dy_gpu1",
    "linux_dy_gpu2",
    # 补充一些自定义标签
    "linux_download_data",
    "linux_gpu1_SST-2",
    "linux_gpu1_QNLI",
    "linux_gpu1_CoLA",
    "linux_gpu1_MRPC",
    "linux_gpu1_STS-B",
    "linux_gpu1_QQP",
    "linux_gpu1_MNLI",
    "linux_gpu1_RTE",
    "linux_finetune_gpu1",
    "linux_gpu1_ernie",
    "linux_gpu1_bigru_crf",
    "linux_gpu2_MNLI",
    "linux_gpu2_RTE",
    "linux_gpu2_QQP",
    "linux_gpu2_STS-B",
    "linux_gpu2_MRPC",
    "linux_gpu2_CoLA",
    "linux_gpu2_QNLI",
    "linux_gpu2_SST-2",
    "linux_gpu1_WNLI",
    "linux_dy_udc_gpu1",
    "linux_dy_udc_gpu2",
    "linux_dy_dstc2_gpu1",
    "linux_dy_dstc2_gpu2",
    "linux_dy_atis_slot_gpu1",
    "linux_dy_atis_slot_gpu2",
    "linux_dy_atis_intent_gpu1",
    "linux_dy_atis_intent_gpu2",
    "linux_dy_mrda_gpu2",
    "linux_dy_mrda_gpu1",
    "linux_dy_swda_gpu2",
    "linux_dy_swda_gpu1",
    "linux_dy_24_gpu1",
    "linux_dy_24_gpu2",
    "linux_dy_32_gpu2",
    "linux_dy_trigger_gpu1",
    "linux_dy_trigger_gpu2",
    "linux_dy_role_gpu1",
    "linux_dy_role_gpu2",
    "linux_dy_enum_gpu1",
    "linux_dy_enum_gpu2",
    "linux_gpu1_ernie_crf",
    "linux_dy_gpu1_batch",
    "linux_dy_gpu2_batch",
    "linux_dy_gpu1_hardest",
    "linux_dy_gpu2_hardest",
    "linux_dy_gpu1_iflytek",
    "linux_dy_gpu1_tnews",
    "linux_dy_gpu1_eprstmt",
    "linux_dy_gpu1_bustm",
    "linux_dy_gpu1_ocnli",
    "linux_dy_gpu1_csl",
    "linux_dy_gpu1_csldcp",
    "linux_dy_gpu1_cluewsc",
    "linux_dy_gpu1_chid",
    "linux_dy_gpu1_point-wise",
    "linux_dy_gpu1_pair-wise",
    "linux_dy_gpu2_point-wise",
    "linux_dy_gpu2_pair-wise",
    "linux_dy_gpu1_chnsenticorp",
    "linux_dy_gpu2_chnsenticorp",
    "linux_dy_gpu1_sst-2",
    "linux_dy_gpu2_sst-2",
    "linux_dy_gpu1_qqp",
    "linux_dy_gpu2_qqp",
    "linux_dy_gpu1_imdb",
    "linux_dy_gpu2_imdb",
    "linux_dy_gpu1_hyp",
    "linux_dy_gpu2_hyp",
    "linux_dy_gpu2_iflytek",
    "linux_dy_gpu1_thucnews",
    "linux_dy_gpu2_thucnews",
    "linux_dy_gpu1_dureader_robust",
    "linux_dy_gpu2_dureader_robust",
    "linux_dy_gpu1_drcd",
    "linux_dy_gpu2_drcd",
    "linux_dy_gpu1_cmrc2018",
    "linux_dy_gpu2_cmrc2018",
    "linux_dy_gpu1_LCQMC",
    "linux_dy_gpu1_BQ_Corpus",
    "linux_dy_gpu1_STS-B",
    "linux_dy_gpu1_ATEC",
    "linux_dy_gpu1_AFQMC",
    "linux_dy_gpu1_TNEWS",
    "linux_dy_gpu1_IFLYTEK",
]
