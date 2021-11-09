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

import pytest
import numpy as np
import subprocess
import re

from RocmTestFramework import TestGanModel
from RocmTestFramework import RepoInit
from RocmTestFramework import RepoRemove
from RocmTestFramework import RepoDataset
from RocmTestFramework import CustomInstruction
from RocmTestFramework import clean_process


def setup_module():
    """
    """
    RepoInit(repo='PaddleGAN')
    RepoDataset(cmd='''cd PaddleGAN; 
                       cd data;
                       rm -rf DIV2K;
                       ln -s /data/DIV2K DIV2K;
                       rm -rf REDS;
                       ln -s /data/REDS REDS;
                       ln -s /data/cityscapes_gan cityscapes;
                       ln -s /data/horse2zebra horse2zebra;
                       cd ..;
                       ln -s /data/gan_model gan_model;
                       python -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple;
                       yum install epel-release -y;
                       yum update -y;
                       rpm --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro;
                       rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm;
                       yum install ffmpeg ffmpeg-devel -y''') 

def teardown_module():
    """
    """
    RepoRemove(repo='PaddleGAN')

def setup_function():
    clean_process()

def test_esrgan_psnr_x4_div2k():
    """
    esrgan_psnr_x4_div2k test case
    """
    model = TestGanModel(model='esrgan_psnr_x4_div2k',
                         yaml='configs/esrgan_psnr_x4_div2k.yaml')
    model.test_gan_train()


def test_edvr_m_wo_tsa():
    """
    edvr_m_wo_tsa test case
    """
    model = TestGanModel(model='edvr_m_wo_tsa',
                         yaml='configs/edvr_m_wo_tsa.yaml')
    model.test_gan_train()

def test_pix2pix_cityscapes():
    """
    pix2pix_cityscapes test case
    """
    model = TestGanModel(model='pix2pix_cityscapes',
                         yaml='configs/pix2pix_cityscapes.yaml')
    cmd='''cd PaddleGAN; python tools/main.py --config-file configs/pix2pix_cityscapes.yaml --evaluate-only --load gan_model/Pix2Pix_cityscapes.pdparams'''
    model.test_gan_eval(cmd=cmd)

def test_cyclegan_horse2zebra():
    """
    cyclegan_horse2zebra test case
    """
    model = TestGanModel(model='cyclegan_horse2zebra',
                         yaml='configs/cyclegan_horse2zebra.yaml')
    cmd='''cd PaddleGAN; python tools/main.py --config-file configs/cyclegan_horse2zebra.yaml --evaluate-only --load gan_model/CycleGAN_horse2zebra.pdparams'''
    model.test_gan_eval(cmd=cmd)


def test_stylegan_v2_256_ffhq():
    """
    stylegan_v2_256_ffhq test case
    """
    model = TestGanModel(model='stylegan_v2_256_ffhq',
                         yaml='configs/stylegan_v2_256_ffhq.yaml')
    cmd='''cd PaddleGAN; python -u applications/tools/styleganv2.py --output_path output_dir/styleganv2 --model_type ffhq-config-f  --seed 233 --size 1024 --style_dim 512 --n_mlp 8 --channel_multiplier 2 --n_row 3 --n_col 5'''

    model.test_gan_eval(cmd=cmd)


def test_wav2lip():
    """
    wav2lip test case
    """
    model = TestGanModel(model='wav2lip',
                         yaml='configs/wav2lip.yaml')
    cmd='''cd PaddleGAN; python -u applications/tools/wav2lip.py --face docs/imgs/mona7s.mp4 --audio docs/imgs/guangquan.m4a --outfile output_dir/pp_guangquan_mona7s.mp4'''
    model.test_gan_eval(cmd=cmd)
