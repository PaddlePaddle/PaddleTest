"""
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""

import os
import time
import tarfile
import logging
import argparse

import wget
import numpy as np

from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.inference import PrecisionType


FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def check_model_exist():
    """
    check model exist
    """
    vgg16_url = "https://paddle-qa.bj.bcebos.com/inference_benchmark/paddle/model/VGG16.tgz"
    if not os.path.exists("./VGG16/inference.pdiparams"):
        wget.download(vgg16_url, out="./")
        tar = tarfile.open("VGG16.tgz")
        tar.extractall()
        tar.close()


def init_predictor(args):
    """
    Args:
        args : input args

    """
    use_calib_mode = False
    if args.trt_precision == "int8":
        use_calib_mode = True

    config = Config("./VGG16/inference.pdmodel", "./VGG16/inference.pdiparams")

    config.enable_memory_optim()
    trt_precision_map = {"fp32": PrecisionType.Float32, "fp16": PrecisionType.Half, "int8": PrecisionType.Int8}
    if args.device == "gpu":
        config.enable_use_gpu(1000, 0)
        if args.use_trt:
            config.enable_tensorrt_engine(
                1 << 30,  # workspace_size
                10,  # max_batch_size
                3,  # min_subgraph_size
                trt_precision_map[args.trt_precision],  # precision
                False,  # use_static
                use_calib_mode,  # use_calib_mode
            )
    elif args.device == "cpu" and args.use_mkldnn:
        config.enable_mkldnn()

    config.set_cpu_math_library_num_threads(4)
    predictor = create_predictor(config)
    return predictor


def run(predictor, img):
    """
    Args:
        predictor : paddle predictor
        img : numpy random array
    """
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(img[i].shape)
        input_tensor.copy_from_cpu(img[i].copy())

    for i in range(args.warmup_times):
        predictor.run()
    time1 = time.time()
    for i in range(args.repeats):
        predictor.run()
    time2 = time.time()
    total_inference_cost = (time2 - time1) * 1000  # total latency, ms

    return total_inference_cost


def parse_args():
    """
    parse args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="batch_size.")
    parser.add_argument("--warmup_times", type=int, default=10, help="warmup_times.")
    parser.add_argument("--repeats", type=int, default=1000, help="repeats.")
    parser.add_argument("--device", type=str, default="gpu", help="[gpu,cpu,xpu]")
    parser.add_argument("--use_trt", type=bool, default=False, help="Whether use trt.")
    parser.add_argument("--trt_precision", type=str, default="fp32", help="Whether use gpu.")
    parser.add_argument(
        "--use_mkldnn", type=int, default=False, help="trt precision, choice = ['fp32', 'fp16', 'int8']"
    )
    return parser.parse_args()


def summary_config(args, infer_time: float):
    """
    Args:
        args : input args
        infer_time : inference time
    """
    logger.info("----------------------- Model info ----------------------")
    logger.info("Model name: {0}, Model type: {1}".format("vgg16", "paddle_model"))
    logger.info("----------------------- Data info -----------------------")
    logger.info("Batch size: {0}, Num of samples: {1}".format(args.batch_size, args.repeats))
    logger.info("----------------------- Conf info -----------------------")
    logger.info("device: {0}".format(args.device))
    if args.use_trt:
        logger.info("enable_tensorrt: {0}".format(args.use_trt))
        logger.info("trt_precision: {0}".format(args.trt_precision))
    logger.info("----------------------- Perf info -----------------------")
    logger.info(
        "Average latency(ms): {0}, QPS: {1}".format(
            infer_time / args.repeats, (args.repeats * args.batch_size) / (infer_time / 1000)
        )
    )


if __name__ == "__main__":
    """
    main case
    """
    check_model_exist()
    args = parse_args()
    pred = init_predictor(args)
    np.random.seed(15)
    img = np.random.randint(0, 255, (args.batch_size, 3, 224, 224)).astype("float32")
    infer_time = run(pred, [img])
    summary_config(args, infer_time)
