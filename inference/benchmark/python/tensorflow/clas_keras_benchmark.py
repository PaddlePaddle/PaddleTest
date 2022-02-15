"""
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""

import argparse
import os
import sys
import time
import logging

import numpy as np
import tensorflow as tf  # tf version should greater than 2.3.0

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
logger.info("==== Tensorflow version: {} ====".format(tf.version.VERSION))


def parse_args():
    """
    parse input args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument(
        "--trt_precision", type=str, default="fp32", help="trt precision, choice = ['fp32', 'fp16', 'int8']"
    )
    parser.add_argument(
        "--image_shape",
        type=str,
        default="3,224,224",
        help="can only use for one input model(e.g. image classification)",
    )

    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")
    parser.add_argument("--use_trt", dest="use_trt", action="store_true")
    parser.add_argument("--use_xla", dest="use_xla", action="store_true")

    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--warmup_times", type=int, default=10, help="warmup")
    parser.add_argument("--repeats", type=int, default=1000, help="repeats")

    return parser.parse_args()


def prepare_model(args):
    """
    prepare tf models from keras
    """
    if args.model_name == "MobileNetV2":
        model = tf.keras.applications.MobileNetV2(weights="imagenet")
        model.save("./{}_saved_model".format(args.model_name))
    elif args.model_name == "VGG16":
        model = tf.keras.applications.VGG16(weights="imagenet")
        model.save("./{}_saved_model".format(args.model_name))
    elif args.model_name == "ResNet101":
        model = tf.keras.applications.ResNet101(weights="imagenet")
        model.save("./{}_saved_model".format(args.model_name))
    else:
        sys.exit(0)

    if args.use_trt and args.trt_precision == "fp32":
        # convert model to trt fp32
        logger.info("Converting to TF-TRT FP32...")
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=trt.TrtPrecisionMode.FP32, max_workspace_size_bytes=8000000000
        )

        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir="{}_saved_model".format(args.model_name), conversion_params=conversion_params
        )
        converter.convert()
        converter.save(output_saved_model_dir="{}_saved_model_TFTRT_FP32".format(args.model_name))
        logger.info("Done Converting to TF-TRT FP32")
    elif args.use_trt and args.trt_precision == "fp16":
        logger.info("Converting to TF-TRT FP16...")
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=trt.TrtPrecisionMode.FP16, max_workspace_size_bytes=8000000000
        )
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir="{}_saved_model".format(args.model_name), conversion_params=conversion_params
        )
        converter.convert()
        converter.save(output_saved_model_dir="{}_saved_model_TFTRT_FP16".format(args.model_name))
        logger.info("Done Converting to TF-TRT FP16")
    elif args.use_trt and args.trt_precision == "int8":
        # convert model to trt int8
        logger.info("Converting to TF-TRT INT8...")
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=trt.TrtPrecisionMode.INT8, max_workspace_size_bytes=8000000000, use_calibration=True
        )
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir="{}_saved_model".format(args.model_name), conversion_params=conversion_params
        )

        channels = int(args.image_shape.split(",")[0])
        height = int(args.image_shape.split(",")[1])
        width = int(args.image_shape.split(",")[2])
        logger.info("channels: {0}, height: {1}, width: {2}".format(channels, height, width))
        input_shape = (args.batch_size, height, width, channels)

        def calibration_input_fn(input_shape):
            batched_input = tf.constant(np.ones(input_shape).astype("float"))
            batched_input = tf.cast(batched_input, dtype="float")
            yield (batched_input,)

        converter.convert(calibration_input_fn=calibration_input_fn(input_shape))
        converter.save(output_saved_model_dir="{}_saved_model_TFTRT_INT8".format(args.model_name))
        logger.info("Done Converting to TF-TRT INT8")
    else:
        logger.warn("No TensorRT precision was input, will not convert TensorRT graph to saved model")


def benchmark_tftrt(args, input_saved_model):
    """
    trt inference
    """
    saved_model_loaded = tf.saved_model.load(input_saved_model, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures["serving_default"]

    channels = int(args.image_shape.split(",")[0])
    height = int(args.image_shape.split(",")[1])
    width = int(args.image_shape.split(",")[2])
    logger.info("channels: {0}, height: {1}, width: {2}".format(channels, height, width))
    input_shape = (args.batch_size, height, width, channels)

    if args.use_gpu:
        run_device = "/gpu:0"
    else:
        run_device = "/cpu:0"
    logger.warn("=== tf.device cannot specify device correctly ===")
    with tf.device(run_device):
        batched_input = tf.constant(np.ones(input_shape).astype("float"))
        batched_input = tf.cast(batched_input, dtype="float")

        for i in range(args.warmup_times):
            infer(batched_input)

        time1 = time.time()
        for i in range(args.repeats):
            infer(batched_input)
        time2 = time.time()
        total_inference_cost = (time2 - time1) * 1000  # total latency, ms

    return total_inference_cost


def summary_config(args, infer_time: float):
    """
    Args:
        args : input args
        infer_time : inference time
    """
    logger.info("----------------------- Model info ----------------------")
    logger.info("Model name: {0}, Model type: {1}".format(args.model_name, "keras_dy2static"))
    logger.info("----------------------- Data info -----------------------")
    logger.info("Batch size: {0}, Num of samples: {1}".format(args.batch_size, args.repeats))
    logger.info("----------------------- Conf info -----------------------")
    logger.info("device: {0}".format("gpu" if args.use_gpu else "cpu"))
    if args.use_gpu:
        logger.info("enable_tensorrt: {0}".format(args.use_trt))
        if args.use_trt:
            logger.info("trt_precision: {0}".format(args.trt_precision))
    logger.info("enable_xla: {0}".format(args.use_xla))
    logger.info("----------------------- Perf info -----------------------")
    logger.info(
        "Average latency(ms): {0}, QPS: {1}".format(
            infer_time / args.repeats, (args.repeats * args.batch_size) / (infer_time / 1000)
        )
    )


def run_demo():
    """
    run_demo
    """
    args = parse_args()
    prepare_model(args)
    if args.use_trt:
        if args.trt_precision == "fp32":
            total_time = benchmark_tftrt(args, "{}_saved_model_TFTRT_FP32".format(args.model_name))
        elif args.trt_precision == "fp16":
            total_time = benchmark_tftrt(args, "{}_saved_model_TFTRT_FP16".format(args.model_name))
        elif args.trt_precision == "int8":
            total_time = benchmark_tftrt(args, "{}_saved_model_TFTRT_INT8".format(args.model_name))
    else:
        total_time = benchmark_tftrt(args, "{}_saved_model".format(args.model_name))
    summary_config(args, total_time)


if __name__ == "__main__":
    run_demo()
