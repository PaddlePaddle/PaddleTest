# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test solov2 model
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
    solov2_url = "https://paddle-qa.bj.bcebos.com/inference_model/2.1.3/detection/solov2.tgz"
    if not os.path.exists("./solov2/model.pdiparams"):
        wget.download(solov2_url, out="./")
        tar = tarfile.open("solov2.tgz")
        tar.extractall()
        tar.close()


def test_config():
    """
    test combined model config
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(model_file="./solov2/model.pdmodel", params_file="./solov2/model.pdiparams")
    test_suite.config_test()


@pytest.mark.win
@pytest.mark.server
@pytest.mark.jetson
@pytest.mark.trt_fp32
def test_trt_fp32_more_bz():
    """
    compared gpu solov2 batch size = [1] outputs with true val
    """
    check_model_exist()

    file_path = "./solov2"
    images_size = 640
    batch_size_pool = [1]
    for batch_size in batch_size_pool:

        test_suite = InferenceTest()
        test_suite.load_config(model_file="./solov2/model.pdmodel", params_file="./solov2/model.pdiparams")
        images_list, images_origin_list, npy_list = test_suite.get_images_npy(
            file_path, images_size, center=False, model_type="det"
        )

        img = images_origin_list[0:batch_size]
        data = np.array(images_list[0:batch_size]).astype("float32")
        scale_factor_pool = []
        for batch in range(batch_size):
            scale_factor = (
                np.array([images_size * 1.0 / img[batch].shape[0], images_size * 1.0 / img[batch].shape[1]])
                .reshape((1, 2))
                .astype(np.float32)
            )
            scale_factor_pool.append(scale_factor)
        scale_factor_pool = np.array(scale_factor_pool).reshape((batch_size, 2))
        im_shape_pool = []
        for batch in range(batch_size):
            im_shape = np.array([images_size, images_size]).reshape((1, 2)).astype(np.float32)
            im_shape_pool.append(im_shape)
        im_shape_pool = np.array(im_shape_pool).reshape((batch_size, 2))
        input_data_dict = {"im_shape": im_shape_pool, "image": data, "scale_factor": scale_factor_pool}
        output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")
        test_suite.load_config(model_file="./solov2/model.pdmodel", params_file="./solov2/model.pdiparams")
        test_suite.trt_more_bz_test(
            input_data_dict, output_data_dict, repeat=1, delta=1e-5, precision="trt_fp32", dynamic=True, tuned=True
        )
        del test_suite
        test_suite = InferenceTest()
        test_suite.load_config(model_file="./solov2/model.pdmodel", params_file="./solov2/model.pdiparams")
        test_suite.trt_more_bz_test(
            input_data_dict, output_data_dict, repeat=1, delta=1e-5, precision="trt_fp32", dynamic=True, tuned=False
        )
