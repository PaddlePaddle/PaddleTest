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
XPU = "cpu"  # 取值gpu或cpu
# 虽是cpu，但cuda值这些留着，不留模版替换变量会出错
# 配置文件中SET_CUDA为None，框架不配置gpu cuda，使用cpu场景
SET_CUDA = "0"
SET_MULTI_CUDA = "0,1"

# PaddleNLP
REPO_PaddleNLP = "https://github.com/PaddlePaddle/PaddleNLP.git"
BASE_BRANCH = "develop"
GPT2_BRANCH = BASE_BRANCH
ELECTRA_BRANCH = BASE_BRANCH
BERT_BRANCH = BASE_BRANCH
EXPRESS_NER_BRANCH = BASE_BRANCH
TRANSFORMER_BRANCH = BASE_BRANCH
XLNET_BRANCH = BASE_BRANCH
GLUE_BRANCH = BASE_BRANCH
SQuAD_BRANCH = BASE_BRANCH
TRANSFORMERXL_BRANCH = BASE_BRANCH
DuReader_yesno_BRANCH = BASE_BRANCH
BIGBIRD_BRANCH = BASE_BRANCH
TCN_BRANCH = BASE_BRANCH
TRANSFORMER_BRANCH = BASE_BRANCH
word_embedding_BRANCH = BASE_BRANCH
waybill_ie_BRANCH = BASE_BRANCH
vae_seq2seq_BRANCH = BASE_BRANCH
text_matching_simnet_BRANCH = BASE_BRANCH
text_matching_finetune_BRANCH = BASE_BRANCH
text_classification_rnn_BRANCH = BASE_BRANCH
text_classification_finetune_BRANCH = BASE_BRANCH
seq2seq_BRANCH = BASE_BRANCH
rnnlm_BRANCH = BASE_BRANCH
msra_ner_BRANCH = BASE_BRANCH
lexical_analysis_BRANCH = BASE_BRANCH
distill_lstm_BRANCH = BASE_BRANCH
TEXT_CLASSIFICATION_PRETRAINED_BRANCH = BASE_BRANCH
DGU_BRANCH = BASE_BRANCH
# linux gpu下， P0的任务要跑的标签 daily
EXEC_PRIORITY = ["p0", "p1"]
EXEC_CASES = ["DATA_PROC", "TRAIN", "INFER"]
EXEC_TAG = [
    "mac_st_cpu",
    "mac_dy_cpu",
    # 自定义的标签
    "mac_download_data",
    "mac_dy_cpu_chnsenticorp",
    "mac_dy_cpu_sst-2",
    "mac_dy_cpu_qqp",
    "mac_dy_cpu_24",
]
