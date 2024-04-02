# !/usr/bin/env python3
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time
import logging

import numpy as np
import paddle.fluid.inference as paddle_infer

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

def parse_args():
    """
    parse input args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument("--model_type", type=str, default="static",
                        help="model generate type")
    parser.add_argument("--model_path", type=str, help="model filename")
    parser.add_argument("--params_path", type=str, default="",
                        help="parameter filename")
    parser.add_argument("--trt_precision", type=str, default="fp32",
                        help="trt precision, choice = ['fp32', 'fp16', 'int8']")
    parser.add_argument("--image_shape", type=str, default="3,224,224",
                        help="can only use for one input model(e.g. image classification)")

    parser.add_argument("--use_gpu", dest="use_gpu", action='store_true')
    parser.add_argument("--use_trt", dest="use_trt", action='store_true')
    parser.add_argument("--use_mkldnn", dest="use_mkldnn", action='store_true')

    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--warmup_times", type=int, default=5, help="warmup")
    parser.add_argument("--repeats", type=int, default=1000, help="repeats")
    parser.add_argument(
        "--cpu_math_library_num_threads",
        type=int,
        default=1,
        help="math_thread_num")
    parser.add_argument("--trt_min_subgraph_size", type=int, default=3,
                        help="tensorrt min_subgraph_size")

    return parser.parse_args()

def prepare_config(args):
    """[summary]
    Args:
        args : input args
    Returns:
        config : paddle inference config
    """
    trt_precision_map = {"fp32" : paddle_infer.PrecisionType.Float32,
                         "fp16" : paddle_infer.PrecisionType.Half,
                         "int8" : paddle_infer.PrecisionType.Int8}
    if (args.params_path != ""):
        logger.info("params_path detected, set model with combined model")
        config = paddle_infer.Config(args.model_path, args.params_path)
    else:
        logger.info("no params_path detected, set model with uncombined model")
        config = paddle_infer.Config(args.model_path)

    if (args.use_gpu or args.use_trt):
        config.enable_use_gpu(100, 0)
        use_calib = True if args.trt_precision == "int8" else False
        if (args.use_trt):
            logger.info("tensorrt enabled")
            config.enable_tensorrt_engine(1 << 30,  # workspace_size
                    args.batch_size,  # max_batch_size
                    args.trt_min_subgraph_size,  # min_subgraph_size
                    trt_precision_map[args.trt_precision],  # Precision precision
                    False,  # use_static
                    use_calib)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(
            args.cpu_math_library_num_threads)
        if (args.use_mkldnn):
            config.enable_mkldnn()
            logger.info("mkldnn enabled")
    config.enable_memory_optim()
    return config


def summary_config(config, args, infer_time : float):
    """
    Args:
        config : paddle inference config
        args : input args
        infer_time : inference time
    """
    logger.info("----------------------- Model info ----------------------")
    logger.info("Model name: {0}, Model type: {1}".format(args.model_name,
                                                          args.model_type))
    logger.info("----------------------- Data info -----------------------")
    logger.info("Batch size: {0}, Num of samples: {1}".format(args.batch_size,
                                                              args.repeats))
    logger.info("----------------------- Conf info -----------------------")
    logger.info("device: {0}, ir_optim: {1}".format("gpu" if config.use_gpu() else "cpu",
                                                    config.ir_optim()))
    # logger.info("enable_memory_optim: {0}".format(config.enable_memory_optim()))
    if (config.use_gpu()):
        logger.info("enable_tensorrt: {0}".format(config.tensorrt_engine_enabled()))
        if (config.tensorrt_engine_enabled()):
            logger.info("trt_precision: {0}".format(args.trt_precision))
    else:
        logger.info("enable_mkldnn: {0}".format(config.mkldnn_enabled()))
        logger.info("cpu_math_library_num_threads: {0}".format(config.cpu_math_library_num_threads()))
    logger.info("----------------------- Perf info -----------------------")
    logger.info("Average latency(ms): {0}, QPS: {1}".format(infer_time / args.repeats,
                                    (args.repeats * args.batch_size)/ (infer_time/1000)))
