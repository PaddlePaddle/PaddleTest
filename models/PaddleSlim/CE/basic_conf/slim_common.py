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
# CE会根据TC的teamcity 再分配
SET_CUDA = "0"
SET_MULTI_CUDA = "0,1"

# PaddleSlim
REPO_PaddleSlim = "https://github.com/PaddlePaddle/PaddleSlim.git"
BASE_BRANCH = "develop"
# 19/23
slim_dy_quant_v1_BRANCH = BASE_BRANCH
slim_dy_pact_quant_v3_BRANCH = BASE_BRANCH
slim_dy_prune_ResNet34_f42_BRANCH = BASE_BRANCH
slim_dy_ofa_TRE_BRANCH = BASE_BRANCH

slim_st_quant_aware_v1_BRANCH = BASE_BRANCH
slim_st_quant_aware_ResNet34_BRANCH = BASE_BRANCH
slim_st_quant_embedding_BRANCH = BASE_BRANCH
slim_st_quant_post_BRANCH = BASE_BRANCH
slim_st_pact_quant_aware_BRANCH = BASE_BRANCH
slim_dy_quant_v3_BRANCH = BASE_BRANCH
slim_dy_ce_tests_qat_BRANCH = BASE_BRANCH
slim_dy_ce_tests_ptq_BRANCH = BASE_BRANCH

slim_st_dist_res50_v1_BRANCH = BASE_BRANCH
slim_st_dist_res101_res50_BRANCH = BASE_BRANCH
slim_st_dist_mv2_x0_25_BRANCH = BASE_BRANCH

slim_st_dml_mv1_mv1_BRANCH = BASE_BRANCH
slim_st_dml_mv1_res50_BRANCH = BASE_BRANCH

slim_st_prune_v1_BRANCH = BASE_BRANCH
slim_st_prune_res50_BRANCH = BASE_BRANCH
slim_st_prune_fpgm_v1_BRANCH = BASE_BRANCH
slim_st_prune_fpgm_v2_BRANCH = BASE_BRANCH
slim_st_prune_fpgm_res34_BRANCH = BASE_BRANCH
slim_dy_prune_fpgm_mobilenetv1_50_BRANCH = BASE_BRANCH
slim_dy_prune_fpgm_mobilenetv2_50_BRANCH = BASE_BRANCH
slim_st_unstructured_prune_threshold_BRANCH = BASE_BRANCH
slim_st_unstructured_prune_ratio_BRANCH = BASE_BRANCH
slim_st_unstructured_prune_mnist_BRANCH = BASE_BRANCH
slim_dy_unstructured_prune_threshold_BRANCH = BASE_BRANCH
slim_dy_unstructured_prune_ratio_BRANCH = BASE_BRANCH
slim_dy_unstructured_prune_cifar10_BRANCH = BASE_BRANCH

slim_st_sa_nas_BRANCH = BASE_BRANCH
slim_st_darts_BRANCH = BASE_BRANCH


# linux下 P0的任务要跑的标签 daily
daily = True
if daily:
    EXEC_PRIORITY = ["p0", "p1"]
    EXEC_CASES = ["DATA_PROC", "TRAIN", "INFER"]
    EXEC_TAG = [
        "linux_st_gpu1",
        "linux_dy_gpu1",
        "linux_st_gpu2",
        "linux_dy_gpu2",
        "linux_st_cpu",
        "linux_dy_cpu",
        # 自定义tag
        "linux_down_data",
    ]
else:
    # linux下 P0的收敛性任务,month
    EXEC_PRIORITY = ["p0", "p1"]
    EXEC_CASES = ["DATA_PROC", "TRAIN", "INFER"]
    EXEC_TAG = ["linux_st_con_gpu2", "linux_dy_con_gpu2", "linux_down_data"]
