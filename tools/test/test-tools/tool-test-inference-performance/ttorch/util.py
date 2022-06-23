# !/usr/bin/env python3
import os
import copy
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as T
import multiprocessing
from multiprocessing import Pool
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
    parser.add_argument("--thread_num", type=int, default=1, help="thread num")
    parser.add_argument("--warmup_turns", type=int, default=5, help="warmup")
    parser.add_argument("--repeats", type=int, default=1000, help="repeats")
    parser.add_argument("--use_gpu", dest="use_gpu", action='store_true')

    return parser.parse_args()


def summary_config(args, total_time: float):
    print("\n\n\n")
    print("----------------------- REPORT START ----------------------")
    print("Model name: {0}".format(args.model_name))
    print("Num of samples: {0}".format(args.repeats))
    print("Thread num: {0}".format(args.thread_num))
    print("Batch size: {0}".format(args.batch_size))
    print("Device: {0}".format("GPU" if args.use_gpu else "CPU"))
    print(f"Average latency(ms): {total_time * 1000 / (args.thread_num * args.repeats)}")
    print(f"QPS: {(args.repeats * args.batch_size * args.thread_num) / total_time}")
    print("------------------------ REPORT END -----------------------")
    print("\n")


class MultiThreadRunner(object):
    def __init__(self):
        multiprocessing.set_start_method('spawn')
        pass

    def run(self, thread_func, thread_num, global_resource):
        resource_list = []
        for i in range(thread_num):
            resource = {
                "predictor": global_resource["predictor_list"][i],
                "input_data": global_resource["input_data_list"][i],
                "repeats": global_resource["repeats"],
            }
            resource_list.append(resource)
        p = Pool(thread_num)
        result_list = []

        start = time.time()
        for i in range(thread_num):
            result_list.append(
                p.apply_async(thread_func, [resource_list[i]]))
        p.close()
        p.join()
        end = time.time()
        total_time = end - start
        return_result = result_list[0].get()

        for i in range(1, thread_num, 1):
            tmp_result = result_list[i].get()
            for i, item in enumerate(tmp_result):
                return_result[i].extend(tmp_result[i])
        return_result.append(total_time)
        #print(return_result)
        return return_result
