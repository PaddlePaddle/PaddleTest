"""
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""

import os
import time
import sys
import argparse
import numpy as np
import cv2
import yaml


import paddle
from backend import PaddleInferenceEngine, TensorRTEngine, ONNXRuntimeEngine, Monitor
from paddle.io import DataLoader
from utils.imagenet_reader import ImageNetDataset


def argsparser():
    """
    argsparser func
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=str, default="./MobileNetV1_infer", help="model directory")
    parser.add_argument("--model_filename", type=str, default="inference.pdmodel", help="model file name")
    parser.add_argument("--params_filename", type=str, default="inference.pdiparams", help="params file name")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--data_path", type=str, default="./dataset/ILSVRC2012_val/")
    parser.add_argument("--use_gpu", type=bool, default=False, help="Whether to use gpu")
    parser.add_argument("--use_mkldnn", type=bool, default=False, help="Whether to use mkldnn")
    parser.add_argument("--cpu_num_threads", type=int, default=10, help="Number of cpu threads")
    parser.add_argument("--precision", type=str, default="paddle", help="mode of running(fp32/fp16/int8)")
    parser.add_argument("--use_trt", type=bool, default=False, help="Whether to use tensorrt")
    parser.add_argument("--gpu_mem", type=int, default=8000, help="GPU memory")
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        help="Choose the device you want to run, it can be: CPU/GPU/XPU, default is GPU",
    )
    parser.add_argument("--cpu_threads", type=int, default=1, help="Num of cpu threads.")
    parser.add_argument("--ir_optim", type=bool, default=True)
    parser.add_argument("--use_dynamic_shape", type=bool, default=True, help="Whether use dynamic shape or not.")
    parser.add_argument("--calibration_file", type=str, default=None, help="quant onnx model calibration cache file.")
    parser.add_argument(
        "--deploy_backend",
        type=str,
        default="paddle_inference",
        help="deploy backend, it can be: `paddle_inference`, `tensorrt`, `onnxruntime`",
    )
    parser.add_argument(
        "--input_name",
        type=str,
        default="x",
        help="input name of image classification model, this is only used by nv-trt",
    )
    parser.add_argument(
        "--small_data",
        action="store_true",
        default=False,
        help="whether val on full data, if not we will only val on 1000 samples",
    )
    parser.add_argument("--model_name", type=str, default="", help="model_name for benchmark")
    return parser


def eval_reader(data_dir, batch_size, crop_size, resize_size):
    """
    eval reader func
    """
    val_reader = ImageNetDataset(mode="val", data_dir=data_dir, crop_size=crop_size, resize_size=resize_size)
    val_loader = DataLoader(val_reader, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    return val_loader


def reader_wrapper(reader, input_field="inputs"):
    """
    reader wrapper func
    """

    def gen():
        for batch_id, (image, label) in enumerate(reader):
            yield np.array(image).astype(np.float32)

    return gen


def eval(predictor, FLAGS):
    """
    eval func
    """
    if os.path.exists(FLAGS.data_path):
        val_loader = eval_reader(
            FLAGS.data_path, batch_size=FLAGS.batch_size, crop_size=FLAGS.img_size, resize_size=FLAGS.resize_size
        )
    else:
        image = np.ones((1, 3, FLAGS.img_size, FLAGS.img_size)).astype(np.float32)
        label = None
        val_loader = [[image, label]]
    results = []
    predict_time = 0.0
    time_min = float("inf")
    time_max = float("-inf")
    warmup = 20
    sample_nums = len(val_loader)
    if FLAGS.small_data:
        sample_nums = 1000

    use_gpu = True
    if FLAGS.device == "CPU":
        use_gpu = False
    monitor = Monitor(0, use_gpu)
    rerun_flag = True if hasattr(predictor, "rerun_flag") and predictor.rerun_flag else False
    # in collect shape mode ,we do not start monitor!
    if not rerun_flag:
        monitor.start()
    for batch_id, (image, label) in enumerate(val_loader):
        image = np.array(image)
        # classfication model usually having only one input
        image = np.expand_dims(image, 0)
        predictor.prepare_data(image)

        for i in range(warmup):
            predictor.run()
            warmup = 0

        start_time = time.time()
        all_output = predictor.run()
        # classfication model usually having only one output
        batch_output = all_output[0]
        end_time = time.time()
        timed = end_time - start_time
        time_min = min(time_min, timed)
        time_max = max(time_max, timed)
        predict_time += timed
        sort_array = batch_output.argsort(axis=1)
        top_1_pred = sort_array[:, -1:][:, ::-1]
        if label is None:
            results.append(top_1_pred)
            break
        label = np.array(label)
        top_1 = np.mean(label == top_1_pred)
        top_5_pred = sort_array[:, -5:][:, ::-1]
        acc_num = 0
        for i, _ in enumerate(label):
            if label[i][0] in top_5_pred[i]:
                acc_num += 1
        top_5 = float(acc_num) / len(label)
        results.append([top_1, top_5])
        if batch_id >= sample_nums:
            break
        if batch_id % 100 == 0:
            print("Eval iter:", batch_id)
            sys.stdout.flush()
        if rerun_flag:
            return

    monitor.stop()
    monitor_result = monitor.output()

    cpu_mem = (
        monitor_result["result"]["cpu_memory.used"]
        if ("result" in monitor_result and "cpu_memory.used" in monitor_result["result"])
        else 0
    )
    gpu_mem = (
        monitor_result["result"]["gpu_memory.used"]
        if ("result" in monitor_result and "gpu_memory.used" in monitor_result["result"])
        else 0
    )

    print("[Benchmark] cpu_mem:{} MB, gpu_mem: {} MB".format(cpu_mem, gpu_mem))

    result = np.mean(np.array(results), axis=0)
    fp_message = FLAGS.precision
    print_msg = "Paddle-Inference-GPU"
    if FLAGS.use_trt and FLAGS.deploy_backend == "paddle_inference":
        print_msg = "using Paddle-TensorRT"
    elif FLAGS.use_mkldnn:
        print_msg = "using Paddle-MKLDNN"
    elif FLAGS.deploy_backend == "tensorrt":
        print_msg = "using NV-TensorRT"
    time_avg = predict_time / sample_nums
    print(
        "[Benchmark]{}\t{}\tbatch size: {}.Inference time(ms): min={}, max={}, avg={}".format(
            print_msg,
            fp_message,
            FLAGS.batch_size,
            round(time_min * 1000, 2),
            round(time_max * 1000, 1),
            round(time_avg * 1000, 1),
        )
    )
    print("[Benchmark] Evaluation acc result: {}".format(result[0]))
    final_res = {
        "model_name": FLAGS.model_name,
        "batch_size": FLAGS.batch_size,
        "jingdu": {
            "value": result[0],
            "unit": "acc",
        },
        "xingneng": {
            "value": round(time_avg * 1000, 1),
            "unit": "ms",
        },
        "gpu_mem": {
            "value": gpu_mem,
            "unit": "MB",
        },
        "cpu_mem": {
            "value": cpu_mem,
            "unit": "MB",
        },
    }
    print("[Benchmark][final result]{}".format(final_res))
    sys.stdout.flush()


def main(FLAGS):
    """
    main func
    """
    predictor = None

    if FLAGS.use_mkldnn:
        FLAGS.device = 'CPU'

    if FLAGS.deploy_backend == "paddle_inference":
        predictor = PaddleInferenceEngine(
            model_dir=FLAGS.model_path,
            model_filename=FLAGS.model_filename,
            params_filename=FLAGS.params_filename,
            precision=FLAGS.precision,
            use_trt=FLAGS.use_trt,
            use_mkldnn=FLAGS.use_mkldnn,
            batch_size=FLAGS.batch_size,
            device=FLAGS.device,
            min_subgraph_size=3,
            use_dynamic_shape=FLAGS.use_dynamic_shape,
            cpu_threads=FLAGS.cpu_threads,
        )
    elif FLAGS.deploy_backend == "tensorrt":

        model_name = os.path.split(FLAGS.model_path)[-1].rstrip(".onnx")
        engine_file = "{}_{}_model.trt".format(model_name, FLAGS.precision)
        print(engine_file)
        predictor = TensorRTEngine(
            onnx_model_file=FLAGS.model_path,
            shape_info={FLAGS.input_name: [[1, 3, 224, 224], [1, 3, 224, 224], [1, 3, 224, 224]]},
            max_batch_size=FLAGS.batch_size,
            precision=FLAGS.precision,
            engine_file_path=engine_file,
            calibration_cache_file=FLAGS.calibration_file,
            calibration_loader=reader_wrapper(
                eval_reader(
                    FLAGS.data_path,
                    batch_size=FLAGS.batch_size,
                    crop_size=FLAGS.img_size,
                    resize_size=FLAGS.resize_size,
                )
            ),
            verbose=False,
        )
    elif FLAGS.deploy_backend == "onnxruntime":
        model_name = os.path.split(FLAGS.model_path)[-1].rstrip(".onnx")
        engine_file = "{}_{}_model.trt".format(model_name, FLAGS.precision)
        print(engine_file)
        predictor = ONNXRuntimeEngine(
            onnx_model_file=FLAGS.model_path,
            precision=FLAGS.precision,
            use_trt=FLAGS.use_trt,
            use_mkldnn=FLAGS.use_mkldnn,
            device=FLAGS.device,
        )
    eval(predictor, FLAGS)
    rerun_flag = True if hasattr(predictor, "rerun_flag") and predictor.rerun_flag else False
    if rerun_flag:
        print("***** Collect dynamic shape done, Please rerun the program to get correct results. *****")
        return


if __name__ == "__main__":
    parser = argsparser()
    FLAGS = parser.parse_args()
    main(FLAGS)
