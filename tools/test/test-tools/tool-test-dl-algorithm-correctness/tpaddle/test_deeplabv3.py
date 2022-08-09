# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test deeplabv3p_resnet50 model
"""

import os
import sys
import logging
import tarfile
import six
import wget
import pytest
import numpy as np
import cv2
import paddle.inference as paddle_infer

# pylint: disable=wrong-import-position
sys.path.append("..")
from test_case import InferenceTest

# pylint: enable=wrong-import-position


def check_model_exist():
    """
    check model exist
    """
    resnet50_url = "https://paddle-qa.bj.bcebos.com/inference_model/2.0/seg/deeplabv3p_resnet50.tgz"
    if not os.path.exists("./deeplabv3p_resnet50/model.pdmodel"):
        wget.download(resnet50_url, out="./")
        tar = tarfile.open("deeplabv3p_resnet50.tgz")
        tar.extractall()
        tar.close()


def preprocess(img_path):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    img = np.array(cv2.imread(img_path)).astype("float32")

    # normalize
    img = img / 255.0
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    std = np.array(std)[np.newaxis, np.newaxis, :]
    img -= mean
    img /= std

    img = np.array([img]).transpose([0, 3, 1, 2])
    return img


def gpu_more_bz_test(self, input_data_dict: dict, output_data_dict: dict, repeat=1, delta=1e-5, gpu_mem=1000):
    """
    test enable_use_gpu()
    Args:
        input_data_dict(dict): input data constructed as dictionary
        output_data_dict(dict): output data constructed as dictionary
        repeat(int): inference repeat time, set to catch gpu mem
        delta(float): difference threshold between inference outputs and thruth value
    Returns:
        None
    """
    self.pd_config.enable_use_gpu(gpu_mem, 0)
    predictor = paddle_infer.create_predictor(self.pd_config)

    input_names = predictor.get_input_names()
    for _, input_data_name in enumerate(input_names):
        input_handle = predictor.get_input_handle(input_data_name)
        input_handle.copy_from_cpu(input_data_dict[input_data_name])

    for i in range(repeat):
        predictor.run()
    output_names = predictor.get_output_names()
    for i, output_data_name in enumerate(output_names):
        output_handle = predictor.get_output_handle(output_data_name)
        output_data = output_handle.copy_to_cpu()
        output_data = output_data.flatten()
        output_data_truth_val = output_data_dict[output_data_name].flatten()
        diff_array = np.abs(output_data - output_data_truth_val)
        diff_count = np.sum(diff_array > delta)
        assert diff_count == 0, f"total: {np.size(diff_array)} diff count:{diff_count} max:{np.max(diff_array)}"


@pytest.mark.p0
@pytest.mark.config_init_combined_model
def test_config():
    """
    test combined model config
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(model_file="./deeplabv3p_resnet50/model.pdmodel", params_file="./deeplabv3p_resnet50/model.pdiparams")
    test_suite.config_test()


def test_gpu_bz1():
    """
    compared trt gpu batch_size=1 deeplabv3p_resnet50 outputs with true val
    """
    check_model_exist()

    file_path = "./deeplabv3p_resnet50"
    images_size = 1024
    batch_size_pool = [1]
    for batch_size in batch_size_pool:
        test_suite = InferenceTest()
        test_suite.load_config(model_file="./deeplabv3p_resnet50/model.pdmodel", params_file="./deeplabv3p_resnet50/model.pdiparams")
        images_list, npy_list = test_suite.get_images_npy(file_path, images_size)

        fake_input = preprocess("./deeplabv3p_resnet50/images/seg_data.png")
        input_data_dict = {"generated_tensor_0": fake_input}
        # output_data_dict = test_suite.get_truth_val(input_data_dict, device="cpu")
        output_data_dict = {"save_infer_model/scale_0.tmp_1": npy_list[0]}

        del test_suite  # destroy class to save memory

        test_suite2 = InferenceTest()
        test_suite2.load_config(model_file="./deeplabv3p_resnet50/model.pdmodel", params_file="./deeplabv3p_resnet50/model.pdiparams")
        gpu_more_bz_test(test_suite2, input_data_dict, output_data_dict, delta=1e-3)

        del test_suite2  # destroy class to save memory
