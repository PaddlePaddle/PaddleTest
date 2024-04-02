# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test maskrcnn model
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
    yolov3_url = "https://paddle-qa.bj.bcebos.com/inference_model/2.0/det/mask_rcnn_r50_1x.tgz"
    if not os.path.exists("./mask_rcnn_r50_1x/model/__model__"):
        wget.download(yolov3_url, out="./")
        tar = tarfile.open("mask_rcnn_r50_1x.tgz")
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
    test_suite.load_config(model_file="./mask_rcnn_r50_1x/model/__model__", params_file="./mask_rcnn_r50_1x/model/__params__")
    test_suite.config_test()


@pytest.mark.p1
@pytest.mark.gpu_more_bz_precision
def test_gpu_bz1():
    """
    compared gpu mask_rcnn_r50_1x batch size = [1] outputs with true val
    """
    check_model_exist()

    file_path = "./mask_rcnn_r50_1x"
    images_size = 640
    batch_size_pool = [1]
    for batch_size in batch_size_pool:

        test_suite = InferenceTest()
        test_suite.load_config(model_file="./mask_rcnn_r50_1x/model/__model__", params_file="./mask_rcnn_r50_1x/model/__params__")
        images_list, images_origin_list, npy_list = test_suite.get_images_npy(
            file_path, images_size, center=False, model_type="det"
        )

        img = images_origin_list[0:batch_size]
        result = npy_list[0: batch_size * 2]
        data = np.array(images_list[0:batch_size]).astype("float32")
        im_shape = np.array([[480, 640, 1]]).astype(np.float32)
        im_info = np.array([[640, 640, 1]]).astype(np.float32)
        input_data_dict = {"im_shape": im_shape, "image": data, "im_info": im_info}

        scale_0 = []
        for batch in range(0, batch_size * 2, 2):
            scale_0 = np.concatenate((scale_0, result[batch].flatten()), axis=0)
        scale_1 = []
        for batch in range(1, batch_size * 2, 2):
            scale_1 = np.concatenate((scale_1, result[batch].flatten()), axis=0)

        output_data_dict = {"save_infer_model/scale_0.tmp_0": scale_0, "save_infer_model/scale_1.tmp_0": scale_1}
        test_suite.load_config(model_file="./mask_rcnn_r50_1x/model/__model__", params_file="./mask_rcnn_r50_1x/model/__params__")
        test_suite.gpu_more_bz_test(input_data_dict, output_data_dict, repeat=1, delta=1e-4)
