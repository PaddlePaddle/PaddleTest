#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
  * @file
  * @author jiaxiao01
  * @date 2021/9/3 3:46 PM
  * @brief clas model inference test case
  *
  **************************************************************************/
"""

import re
import subprocess
import pytest
import numpy as np

from RocmTestFramework import TestVideoModel
from RocmTestFramework import RepoInitSummer
from RocmTestFramework import RepoInitCustom
from RocmTestFramework import RepoInitCustom_video
from RocmTestFramework import RepoInit
from RocmTestFramework import RepoRemove
from RocmTestFramework import RepoDataset
from RocmTestFramework import clean_process


def setup_module():
    """
    function
    """
    RepoInit(repo="PaddleVideo")
    RepoInit(repo="models")
    # RepoInitSummer(repo='PaddleVideo')
    # RepoInitCustom_video(repo='models')
    RepoDataset(
        cmd=""" cd PaddleVideo; \
                python -m pip install decord -i https://pypi.tuna.tsinghua.edu.cn/simple; \
                python -m pip install paddlenlp  -i https://pypi.tuna.tsinghua.edu.cn/simple; \
                python -m pip install SimpleITK -i https://pypi.tuna.tsinghua.edu.cn/simple; \
                python -m pip install lmdb -i https://pypi.tuna.tsinghua.edu.cn/simple; \
                cd data; rm -rf ucf101; ln -s /data/ucf101 ucf101; ln -s /data/bmn_data bmn_data; \
                rm -rf TSM_k400.pdparams; \
                wget -q https://videotag.bj.bcebos.com/PaddleVideo-release2.1/TSM/TSM_k400.pdparams; \
                mkdir dataset; \
                cd dataset; \
                ln -s /data/bmn_data bmn_data; cd ..; \
                wget -q https://videotag.bj.bcebos.com/PaddleVideo/SlowFast/SlowFast.pdparams; \
                wget -q https://videotag.bj.bcebos.com/PaddleVideo-release2.2/TSN_k400.pdparams; \
                rm -rf k400; \
                ln -s /data/kinetics k400"""
    )
    RepoDataset(
        cmd=""" cd models/PaddleCV/video; \
                cd data; \
                rm -rf AttentionLSTM.pdparams \
                wget -q https://paddlemodels.bj.bcebos.com/video_classification/AttentionLSTM.pdparams; \
                wget -q https://paddlemodels.bj.bcebos.com/video_classification/STNET.pdparams; \
                cd ..; \
                cd data/dataset; \
                rm -rf youtube8m; \
                rm -rf kinetics; \
                ln -s /data/youtube8m youtube8m; \
                ln -s /data/kinetics kinetics; \
                cd ../..; \
                python -m pip install wget -q -i https://pypi.tuna.tsinghua.edu.cn/simple"""
    )


def teardown_module():
    """
    function
    """
    RepoRemove(repo="PaddleVideo")
    RepoRemove(repo="models")


def setup_function():
    """
    function
    """
    clean_process()


def test_tsm():
    """
    function
    """
    model = TestVideoModel(model="tsm", yaml="configs/recognition/tsm/tsm_ucf101_frames.yaml")
    model.test_video_train()


def test_bmn():
    """
    function
    """
    model = TestVideoModel(model="bmn", yaml="configs/localization/bmn.yaml")
    model.test_video_train()


def test_slowfast():
    """
    function
    """
    model = TestVideoModel(model="slowfast", yaml="configs/recognition/slowfast/slowfast.yaml")
    cmd = """cd PaddleVideo; python -B -m paddle.distributed.launch \
            --gpus="0,1,2,3" --log_dir=log_slowfast_test main.py \
            --test -c  configs/recognition/slowfast/slowfast.yaml \
            -w data/SlowFast.pdparams -o DATASET.test.file_path=data/k400/val_small_change.list.list \
            -o DATASET.test.data_prefix=data/k400"""
    model.test_video_eval(cmd=cmd)


def test_tsn():
    """
    function
    """
    model = TestVideoModel(model="tsn", yaml="configs/recognition/tsn/tsn_k400_frames.yaml")
    cmd = """cd PaddleVideo; python3.7 -B -m paddle.distributed.launch \
            --gpus="0,1,2,3" --log_dir=log_tsn main.py  \
            --test -c configs/recognition/tsn/tsn_k400_videos.yaml  \
            -w "data/TSN_k400.pdparams" -o DATASET.test.format=VideoDataset \
            -o DATASET.test.file_path=data/k400/val_small.list \
            -o DATASET.test.suffix="" -o PIPELINE.test.decode.name=VideoDecoder"""
    model.test_video_eval(cmd=cmd)


def test_attention_lstm():
    """
    function
    """
    model = TestVideoModel(model="attention_lstm", yaml="./configs/attention_lstm.yaml")
    cmd = """cd models/PaddleCV/video;  \
                sed -i s/test.list/test_small.list/g configs/attention_lstm.yaml; \
                python eval.py --model_name=AttentionLSTM \
                --config=./configs/attention_lstm.yaml \
                --log_interval=1 --weights=data/AttentionLSTM.pdparams --use_gpu=True"""
    model.test_video_eval(cmd=cmd)


def test_stnet():
    """
    function
    """
    model = TestVideoModel(model="stnet", yaml="./configs/stnet.yaml")
    cmd = """cd models/PaddleCV/video; \
                sed -i s/pkl/mp4/g configs/stnet.yaml; \
                sed -i s/test.list/test_small.list/g ./configs/stnet.yaml; \
                python eval.py --model_name=STNET --config=./configs/stnet.yaml \
                --log_interval=1 --weights=data/STNET.pdparams --use_gpu=True"""
    model.test_video_eval(cmd=cmd)
