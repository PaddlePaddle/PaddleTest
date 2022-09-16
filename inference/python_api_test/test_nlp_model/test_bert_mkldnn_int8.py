# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test bert model
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
    bert_url = "https://paddle-qa.bj.bcebos.com/inference_model/2.1.2/nlp/bert.tgz"
    if not os.path.exists("./bert/inference.pdiparams"):
        wget.download(bert_url, out="./")
        tar = tarfile.open("bert.tgz")
        tar.extractall()
        tar.close()


def test_config():
    """
    test combined model config
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(model_file="./bert/inference.pdmodel", params_file="./bert/inference.pdiparams")
    test_suite.config_test()


@pytest.mark.win
@pytest.mark.server
@pytest.mark.mkldnn_int8
def test_mkldnn_int8():
    """
    compared mkldnn_int8 bert outputs with true val
    """
    check_model_exist()

    test_suite = InferenceTest()
    test_suite.load_config(model_file="./bert/inference.pdmodel", params_file="./bert/inference.pdiparams")
    data_path = "./bert/data.txt"
    images_list = test_suite.get_text_npy(data_path)

    input_data_dict = {
        "input_ids": np.array([images_list[0][0]]).astype("int64"),
        "token_type_ids": np.array([images_list[0][1]]).astype("int64"),
    }
    output_data_dict = test_suite.get_truth_val(input_data_dict, device="cpu")

    del test_suite  # destroy class to save memory

    test_suite2 = InferenceTest()
    test_suite2.load_config(model_file="./bert/inference.pdmodel", params_file="./bert/inference.pdiparams")
    test_suite2.mkldnn_test(input_data_dict, output_data_dict, delta=1e-5, precision="int8")

    del test_suite2  # destroy class to save memory
