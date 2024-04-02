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

import test_helper as helper


def Inference(args, predictor) -> int:
    """
    paddle-inference
    Args:
        args : python input arguments
        predictor : paddle-inference predictor
    Returns:
        total_inference_cost (int): inference time
    """
    channels = int(args.image_shape.split(',')[0])
    height = int(args.image_shape.split(',')[1])
    width = int(args.image_shape.split(',')[2])

    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    fake_input = np.ones((args.batch_size, channels, height, width)).astype("float32")
    input_handle.reshape([args.batch_size, channels, height, width])
    input_handle.copy_from_cpu(fake_input)

    for i in range(args.warmup_times):
        predictor.run()
        output_names = predictor.get_output_names()
        output_hanlde = predictor.get_output_handle(output_names[0])
        output_data = output_hanlde.copy_to_cpu()

    time1 = time.time()
    for i in range(args.repeats):
        predictor.run()
        output_names = predictor.get_output_names()
        output_hanlde = predictor.get_output_handle(output_names[0])
        output_data = output_hanlde.copy_to_cpu()
    time2 = time.time()
    total_inference_cost = (time2 - time1) * 1000  # total latency, ms

    return total_inference_cost


def run_demo():
    """
    run_demo
    """
    args = helper.parse_args()
    config = helper.prepare_config(args)
    predictor_pool = paddle_infer.PredictorPool(config, 1)
    predictor = predictor_pool.retrive(0)
    total_time = Inference(args, predictor)

    helper.summary_config(config, args, total_time)

if __name__ == "__main__":
    run_demo()
