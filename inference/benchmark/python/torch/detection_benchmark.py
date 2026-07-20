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
import wget
import numpy as np
import torch
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
        if args.model_name == "faster_rcnn":
            self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()
        elif args.model_name == "yolov3":
            yolov3_url = "https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3.pt"
            if not os.path.exists("yolov3.pt"):
                wget.download(yolov3_url, out="./")
            self.model = torch.load("yolov3.pt")
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
    parser.add_argument("--model_name", type=str, default="yolov3", choices=["yolov3", "faster_rcnn"])
    parser.add_argument("--device", default="gpu", type=str, choices=["gpu", "cpu"])
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--warmup_times", type=int, default=10, help="warmup")
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
    img = np.random.randint(0, 255, (args.batch_size, 3, 640, 640)).astype("float32")
    # input_data = np.random.randint(0, 256, size=(args.batch_size, 3, 224, 224), dtype=np.float32)
    image_tensor = torch.from_numpy(img).to(device)
    # image_tensor = torch.randn((args.batch_size, 3, 224, 224)).to("cuda")
    # set running device on
    predictor = Predictor().to(device)
    # predictor = torch.jit.script(predictor).to(device)
    print(image_tensor.dtype)
    logger.info("input image tensor shape : {}".format(image_tensor.shape))

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
    total_time = forward_benchmark(args)[0]
    summary_config(args, total_time)


if __name__ == "__main__":
    run_demo()
