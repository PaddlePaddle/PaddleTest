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


@pytest.mark.p0
@pytest.mark.config_init_combined_model
def test_config():
    """
    test combined model config
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(model_path="./resnet50_quant/resnet50_quant")
    test_suite.config_test()


@pytest.mark.p0
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


@pytest.mark.p1
@pytest.mark.trt_fp32_more_bz_precision
def test_int8_more_bz():
    """
    compared trt fp32 batch_size=1-10 resnet50 outputs with true val
    """
    check_model_exist()

    file_path = "./resnet50_quant"
    images_size = 224
    batch_size_pool = [1, 5, 10]
    max_batch_size = 10
    for batch_size in batch_size_pool:
        test_suite = InferenceTest()
        if 'win' in sys.platform:
            test_suite.load_config(model_path=".\\resnet50_quant\\resnet50_quant")
        else:
            test_suite.load_config(model_path="./resnet50_quant/resnet50_quant")
        images_list, npy_list = test_suite.get_images_npy(file_path, images_size)
        fake_input = np.array(images_list[0:batch_size]).astype("float32")
        fake_output = np.array(npy_list[0:batch_size]).astype("float32")
        input_data_dict = {"image": fake_input}
        output_data_dict = {"save_infer_model/scale_0.tmp_0": fake_output}
        test_suite.trt_more_bz_test(
            input_data_dict,
            output_data_dict,
            delta=5e-1,
            max_batch_size=max_batch_size,
            precision="trt_int8",
            use_calib_mode=True,
        )

        del test_suite  # destroy class to save memory
