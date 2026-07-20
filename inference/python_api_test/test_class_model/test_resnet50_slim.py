# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test resnet50 model
"""

import os
import sys
import logging
import tarfile
import six
import wget
import pytest
import numpy as np

# pylint: disable=wrong-import-position
sys.path.append("..")
from test_case import InferenceTest

# pylint: enable=wrong-import-position


def check_model_exist():
    """
    check model exist
    """
    resnet50_slim_url = "https://paddle-qa.bj.bcebos.com/inference_model/unknown/resnet50_quant.tgz"
    if not os.path.exists("./resnet50_quant/resnet50_quant/__model__"):
        wget.download(resnet50_slim_url, out="./")
        tar = tarfile.open("resnet50_quant.tgz")
        tar.extractall()
        tar.close()


def test_config():
    """
    test combined model config
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(model_path="./resnet50_quant/resnet50_quant")
    test_suite.config_test()


@pytest.mark.win
@pytest.mark.server
@pytest.mark.config_disablegpu_memory
def test_disable_gpu():
    """
    test no gpu resources occupied after disable gpu
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(model_path="./resnet50_quant/resnet50_quant")
    batch_size = 1
    fake_input = np.random.randn(batch_size, 3, 224, 224).astype("float32")
    input_data_dict = {"image": fake_input}
    test_suite.disable_gpu_test(input_data_dict)
