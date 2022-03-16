# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test ocr_structext model
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
from test_case import InferenceTest, clip_model_extra_op

# pylint: enable=wrong-import-position


def check_model_exist():
    """
    check model exist
    """
    ocr_det_mv3_db_url = "https://paddle-qa.bj.bcebos.com/inference_model/2.1/ocr/StrucText.tgz"
    if not os.path.exists("./StrucText/model"):
        wget.download(ocr_det_mv3_db_url, out="./")
        tar = tarfile.open("StrucText.tgz")
        tar.extractall()
        tar.close()
        clip_model_extra_op(model_dir="./StrucText")


def test_config():
    """
    test combined model config
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(model_file="./StrucText/model", params_file="./StrucText/params")
    test_suite.config_test()


@pytest.mark.win
@pytest.mark.server
@pytest.mark.mkldnn
def test_mkldnn():
    """
    compared mkldnn batch_size=1 StrucText outputs with true val
    """
    check_model_exist()

    images_size = 512
    batch_size_pool = [1]
    for batch_size in batch_size_pool:
        test_suite = InferenceTest()
        test_suite.load_config(model_file="./StrucText/model", params_file="./StrucText/params")
        input_data_dict = {}
        input_data_dict["images"] = np.ones((batch_size, 3, images_size, images_size)).astype(np.float32)
        input_data_dict["seq_token"] = np.ones((batch_size, images_size)).astype(np.int64)
        input_data_dict["seq_ids"] = np.ones((batch_size, images_size)).astype(np.int64)
        input_data_dict["seq_mask"] = np.ones((batch_size, images_size)).astype(np.int64)
        input_data_dict["seq_bboxes"] = np.ones((batch_size, images_size, 4)).astype(np.float32)
        output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

        del test_suite  # destroy class to save memory

        test_suite2 = InferenceTest()
        test_suite2.load_config(model_file="./StrucText/model", params_file="./StrucText/params")
        test_suite2.pd_config.enable_memory_optim()

        test_suite2.mkldnn_test(input_data_dict, output_data_dict, repeat=1, delta=2e-5)

        del test_suite2  # destroy class to save memory
