"""
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
import argparse
import logging
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch_tensorrt
import torchvision.models as models

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


class Predictor(torch.nn.Module):
    """
    python inference model
    """

    def __init__(self):
        """
        model name
        """
        super().__init__()

        args = parse_args()
        if args.model_name == "resnet50":
            self.model = models.alexnet(pretrained=True).eval()
        elif args.model_name == "resnet101":
            self.model = models.resnet101(pretrained=True).eval()
        elif args.model_name == "alexnet":
            self.model = models.alexnet(pretrained=True).eval()
        elif args.model_name == "vgg16":
            self.model = models.vgg16(pretrained=True).eval()
        elif args.model_name == "squeezenet1_0":
            self.model = models.squeezenet1_0(pretrained=True).eval()
        elif args.model_name == "inception_v3":
            self.model = models.inception_v3(pretrained=True).eval()
        elif args.model_name == "mobilenet_v2":
            self.model = models.mobilenet_v2(pretrained=True).eval()
        else:
            raise Exception(
                "net type [%s] invalid! \
                        \n please specify corret model_name"
                % args.model_name
            )

    def forward(self, x):
        """
        model forward inference
        Args:
            x: input
        Returns:
            y_pred: output
        """
        with torch.no_grad():
            y_pred = self.model(x)
            return y_pred


def parse_args():
    """
    Args input
    Returns: Args
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        choices=["resnet50", "resnet101", "alexnet", "vgg16", "squeezenet1_0", "inception_v3", "mobilenet_v2"],
    )
    parser.add_argument(
        "--trt_precision", type=str, default="fp32", help="trt precision, choice = ['fp32', 'fp16', 'int8']"
    )
    parser.add_argument("--device", default="gpu", type=str, choices=["gpu", "cpu"])
    parser.add_argument("--use_trt", dest="use_trt", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--warmup_times", type=int, default=5, help="warmup")
    parser.add_argument("--repeats", type=int, default=1000, help="repeats")
    return parser.parse_args()


def forward_benchmark(args):
    """
    forward inference
    Args:
        args
    Returns:
        infernce benchmark time
    """
    if args.device == "gpu":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    np.random.seed(15)
    img = np.random.randint(0, 255, (args.batch_size, 3, 224, 224)).astype("float32")
    # input_data = np.random.randint(0, 256, size=(args.batch_size, 3, 224, 224), dtype=np.float32)
    image_tensor = torch.from_numpy(img).to(device)
    # image_tensor = torch.randn((args.batch_size, 3, 224, 224)).to("cuda")
    print(image_tensor.dtype)
    logger.info("input image tensor shape : {}".format(image_tensor.shape))
    # set running device on
    predictor = Predictor()
    image_tensor = image_tensor.to(device)
    predictor = torch.jit.script(predictor).to(device)
    with torch.no_grad():
        # warm up
        for i in range(args.warmup_times):
            output = predictor(image_tensor)

        time1 = time.time()
        for i in range(args.repeats):
            output = predictor(image_tensor)
        time2 = time.time()
        total_inference_cost = (time2 - time1) * 1000  # total latency, ms
    return total_inference_cost, output


def trt_benchmark(args):
    """
    trt forward inference
    Args:
        args
    Returns:
        infernce trt benchmark time
    """

    # Compile module
    predictor = Predictor()
    device = torch.device("cuda:0")
    image_tensor = torch.randn((1, 3, 224, 224)).to(device)
    # Trace the module with example data
    traced_model = torch.jit.trace(predictor.to(device), [image_tensor]).to(device)

    # Compile module
    compiled_trt_model = torch_tensorrt.compile(
        traced_model,
        inputs=[torch_tensorrt.Input(image_tensor.shape)],
        enabled_precisions={torch.float32},  # Run in FP32
    )
    for i in range(args.warmup_times):
        results = compiled_trt_model(image_tensor)
    time1 = time.time()
    for i in range(args.repeats):
        results = compiled_trt_model(image_tensor)
    time2 = time.time()
    total_inference_cost = (time2 - time1) * 1000  # total latency, ms
    return total_inference_cost, results


def summary_config(args, infer_time: float):
    """
    Args:
        args : input args
        infer_time : inference time
    """
    logger.info("----------------------- Model info ----------------------")
    logger.info("Model name: {0}, Model type: {1}".format(args.model_name, "torch_model"))
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


def run_demo():
    """
    run_demo
    """
    args = parse_args()
    if args.use_trt:
        total_time = trt_benchmark(args)[0]
    else:
        total_time = forward_benchmark(args)[0]
    summary_config(args, total_time)


if __name__ == "__main__":
    run_demo()
