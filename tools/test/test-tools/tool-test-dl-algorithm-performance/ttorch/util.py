# !/usr/bin/env python3
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torchvision.io import read_image
from PIL import Image


FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def image_to_tensor(path: str):
    return T.ToTensor()(Image.open(path))


def batch_input(sample: torch.Tensor, batch_size: int = 1):
    return torch.stack([sample for _ in range(batch_size)])


def check_result(result_data: torch.Tensor, truth_data: torch.Tensor, delta=1e-3):
    result_data = np.array(result_data.to("cpu"))
    truth_data = np.array(truth_data.to("cpu"))

    diff_array = np.abs(result_data - truth_data)
    diff_count = np.sum(diff_array > delta)
    assert diff_count == 0, f"total: {np.size(diff_array)} diff count:{diff_count} max:{np.max(diff_array)}"


def inference(predictor: nn.Module, x: torch.Tensor, args):
    batch_size = int(args.batch_size)
    warmup = int(args.warmup_turns)
    repeats = int(args.repeats)

    batch = torch.stack([x for _ in range(batch_size)])

    for i in range(warmup):
        output = predictor(batch)

    start_time = time.time()
    for i in range(repeats):
        output = predictor(batch)
        output.to("cpu")
    end_time = time.time()
    total_time = (end_time - start_time) * 1000
    return total_time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--warmup_turns", type=int, default=5, help="warmup")
    parser.add_argument("--repeats", type=int, default=1000, help="repeats")
    parser.add_argument("--use_gpu", dest="use_gpu", action='store_true')

    return parser.parse_args()


def summary_config(args, infer_time: float):
    logger.info("----------------------- Model info ----------------------")
    logger.info("Model name: {0}".format(args.model_name))
    logger.info("----------------------- Data info -----------------------")
    logger.info("Batch size: {0}, Num of samples: {1}".format(args.batch_size,
                                                              args.repeats))
    logger.info("----------------------- Conf info -----------------------")
    logger.info("device: {0}".format("gpu" if args.use_gpu else "cpu"))
    logger.info("----------------------- Perf info -----------------------")
    logger.info("Average latency(ms): {0}, QPS: {1}".format(infer_time / args.repeats,
                                    (args.repeats * args.batch_size)/ (infer_time/1000)))
