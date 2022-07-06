# !/usr/bin/env python3
import copy
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torchvision.io import read_image
from PIL import Image

from util import *


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg16 = models.vgg16(pretrained=True).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y_pred = self.vgg16(x)
            return y_pred


def single_func(resource):
    predictor = resource["predictor"]
    input_data = resource["input_data"]
    repeats = resource["repeats"]

    start_time = time.time()
    for i in range(repeats):
        output = predictor(input_data)
        output.to("cpu")
    end_time = time.time()
    total_time = (end_time - start_time) * 1000
    return [[total_time]]


def run_demo(args):
    # 1.prepare
    device = "cuda" if args.use_gpu else "cpu"
    batch_size = args.batch_size
    warmup_turns = args.warmup_turns
    thread_num = args.thread_num
    repeats = args.repeats

    transforms = nn.Sequential(
        T.Resize([256, ]),
        T.CenterCrop(224),
        T.ConvertImageDtype(torch.float),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )
    image_tensor = transforms(image_to_tensor("daisy.jpg"))
    batch = torch.stack([image_tensor for _ in range(batch_size)])

    # 2.to cuda
    batch_list = [batch.to(device) for _ in range(thread_num)]
    predictor_list = [Predictor().to(device) for _ in range(thread_num)]

    # 3.warm up
    for i in range(thread_num):
        for _ in range(warmup_turns):
            output = predictor_list[i](batch_list[i])

    global_resource = {
        "predictor_list": predictor_list,
        "input_data_list": batch_list,
        "repeats": repeats,
    }

    # 4.multi thread run
    multi_thread_runner = MultiThreadRunner()

    result = multi_thread_runner.run(single_func, thread_num, global_resource)

    total_time = result[1]
    #print(total_time)

    summary_config(args, total_time)


if __name__ == '__main__':
    args = parse_args()
    run_demo(args)
