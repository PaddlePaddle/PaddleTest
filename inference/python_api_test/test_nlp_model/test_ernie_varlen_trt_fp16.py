# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test ernie_model_4 model
"""

import os
import sys
import logging
import tarfile
import wget
import pytest
import numpy as np
import paddle.inference as paddle_infer

from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.inference import PrecisionType
from paddle.inference import InternalUtils

# pylint: disable=wrong-import-position
sys.path.append("..")
from test_case import InferenceTest
from test_case.image_preprocess import sig_fig_compare

# pylint: enable=wrong-import-position


def check_model_exist():
    """
    check model exist
    """
    ernie_model_4_url = "https://paddle-qa.bj.bcebos.com/inference_model/unknown/nlp/ernie_model_4.tgz"
    if not os.path.exists("./ernie_model_4/__model__"):
        wget.download(ernie_model_4_url, out="./")
        tar = tarfile.open("ernie_model_4.tgz")
        tar.extractall()
        tar.close()


def test_config():
    """
    test combined model config
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(model_path="./ernie_model_4")
    test_suite.config_test()


def init_predictor(model_path):
    """
    Args:
        model_path (str): Path to the TensorRT model
    Returns:
        Predictor: Returns a TensorRT model predictor object
    """
    config = Config(model_path)
    config.enable_memory_optim()
    config.enable_use_gpu(1000, 0, PrecisionType.Float32)

    config.enable_tensorrt_engine(
        workspace_size=1 << 30,
        max_batch_size=1,
        min_subgraph_size=5,
        precision_mode=PrecisionType.Half,
        use_static=False,
        use_calib_mode=False,
    )

    min_batch = 1
    max_batch = 10
    min_single_seq_len = 1
    max_single_seq_len = 384
    opt_single_seq_len = 384
    min_batch_seq_len = 1
    max_batch_seq_len = 3840
    opt_batch_seq_len = 3840

    input_name0 = "read_file_0.tmp_0"
    input_name1 = "read_file_0.tmp_1"
    input_name2 = "read_file_0.tmp_2"
    input_name3 = "read_file_0.tmp_4"

    min_shape = [min_batch_seq_len]
    max_shape = [max_batch_seq_len]
    opt_shape = [opt_batch_seq_len]

    config.set_trt_dynamic_shape_info(
        {
            input_name0: min_shape,
            input_name1: min_shape,
            input_name2: [1],
            input_name3: [min_batch, min_single_seq_len, 1],
        },
        {
            input_name0: max_shape,
            input_name1: max_shape,
            input_name2: [max_batch + 1],
            input_name3: [max_batch, max_single_seq_len, 1],
        },
        {
            input_name0: opt_shape,
            input_name1: opt_shape,
            input_name2: [max_batch + 1],
            input_name3: [max_batch, opt_single_seq_len, 1],
        },
    )

    config.enable_tensorrt_varseqlen()
    InternalUtils.set_transformer_posid(config, input_name2)
    InternalUtils.set_transformer_maskid(config, input_name3)

    predictor = create_predictor(config)
    return predictor


def run(predictor, delta):
    """
    Runs model prediction and compares the prediction results with the real values within a tolerance threshold.
    Args:
        predictor (Predictor): A model predictor object.
        delta (float): A tolerance threshold for comparison.
    Returns:
        None
    """
    run_batch = 10
    seq_len = 384
    run_seq_len = run_batch * seq_len
    max_seq_len = seq_len
    i0 = np.ones(run_seq_len, dtype=np.int64)
    i1 = np.zeros(run_seq_len, dtype=np.int64)
    i2 = np.array([0, 384, 768, 1152, 1536, 1920, 2304, 2688, 3072, 3456, 3840], dtype=np.int64)
    i3 = np.ones([run_batch, max_seq_len, 1], dtype=float)

    input_names = predictor.get_input_names()

    input_tensor0 = predictor.get_input_handle(input_names[0])
    input_tensor0.copy_from_cpu(i0)

    input_tensor1 = predictor.get_input_handle(input_names[1])
    input_tensor1.copy_from_cpu(i1)

    input_tensor2 = predictor.get_input_handle(input_names[2])
    input_tensor2.copy_from_cpu(i2)

    input_tensor3 = predictor.get_input_handle(input_names[3])
    input_tensor3.copy_from_cpu(i3)

    # do the inference
    predictor.run()

    # get out data from output tensor
    output_names = predictor.get_output_names()

    output_data_dict = {}
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        output_data_dict[name] = output_data

    output_data_array = np.array(output_data_dict["save_infer_model/scale_0"])

    if "win" in sys.platform:
        npy_path = ".\\ernie_model_4\\output_data_truth_val.npy"
    else:
        npy_path = "./ernie_model_4/output_data_truth_val.npy"

    output_data_truth_val = np.load(npy_path, allow_pickle=True)
    truth_val_dict = output_data_truth_val.item()
    truth_val_array = truth_val_dict["save_infer_model/scale_0"]
    diff = sig_fig_compare(output_data_array, truth_val_array, delta)


# skip test on trt_ver < 7.2 platform
ver = paddle_infer.get_trt_compile_version()
if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 7200:
    trt_skip = pytest.mark.skip(reason="Varlen only support trt_fp16, trt_int8 on trt_ver > 7.2")
else:
    trt_skip = pytest.mark.none


@pytest.mark.win
@pytest.mark.server
@pytest.mark.trt_fp16
@trt_skip
def test_trt_fp16():
    """
    compared trt_fp16 ernie_varlen outputs with true val
    """
    check_model_exist()

    if "win" in sys.platform:
        model_path = ".\\ernie_model_4"
    else:
        model_path = "./ernie_model_4"

    pred = init_predictor(model_path)
    run(pred, delta=1e-3)
