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

import argparse
import time
import os
import sys
import cv2
import numpy as np
import paddle
from paddleseg.cvlibs import Config as PaddleSegDataConfig
from paddleseg.core.infer import reverse_transform
from paddleseg.utils import metrics

from backend import PaddleInferenceEngine, TensorRTEngine, ONNXRuntimeEngine


def argsparser():
    """
    argsparser func
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="inference model filepath")
    parser.add_argument("--model_filename", type=str, default="model.pdmodel", help="model file name")
    parser.add_argument("--params_filename", type=str, default="model.pdiparams", help="params file name")
    parser.add_argument(
        "--dataset",
        type=str,
        default="human",
        choices=["human", "cityscape"],
        help="The type of given image which can be 'human' or 'cityscape'.",
    )
    parser.add_argument(
        "--deploy_backend",
        type=str,
        default="paddle_inference",
        help="deploy backend, it can be: `paddle_inference`, `tensorrt`, `onnxruntime`",
    )
    parser.add_argument("--dataset_config", type=str, default=None, help="path of dataset config.")
    parser.add_argument("--benchmark", type=bool, default=False, help="Whether to run benchmark or not.")
    parser.add_argument("--use_trt", type=bool, default=False, help="Whether to use tensorrt engine or not.")
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        choices=["CPU", "GPU"],
        help="Choose the device you want to run, it can be: CPU/GPU, default is GPU",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "int8", "bf16"],
        help=("The precision of inference. It can be 'fp32', 'fp16', 'int8' or 'bf16'."),
    )
    parser.add_argument("--use_mkldnn", type=bool, default=False, help="Whether use mkldnn or not.")
    parser.add_argument("--cpu_threads", type=int, default=1, help="Num of cpu threads.")
    parser.add_argument("--calibration_file", type=str, default=None, help="quant onnx model calibration cache file.")
    parser.add_argument("--model_name", type=str, default="", help="model_name for benchmark")
    parser.add_argument("--small_data", action="store_true", default=False, help="Whether use small data to eval.")
    return parser


def eval(predictor, loader, eval_dataset, rerun_flag):
    """
    eval mIoU func
    """
    intersect_area_all = 0
    pred_area_all = 0
    label_area_all = 0

    predict_time = 0.0
    time_min = float("inf")
    time_max = float("-inf")
    warmup = 20
    print("Start evaluating (total_samples: {}, total_iters: {}).".format(FLAGS.total_samples, FLAGS.sample_nums))
    for batch_id, data in enumerate(loader):
        image = np.array(data[0])
        label = np.array(data[1]).astype("int64")
        ori_shape = np.array(label).shape[-2:]

        predictor.prepare_data([image])

        for i in range(warmup):
            predictor.run()
            warmup = 0

        start_time = time.time()
        outs = predictor.run()
        end_time = time.time()

        timed = end_time - start_time
        time_min = min(time_min, timed)
        time_max = max(time_max, timed)
        predict_time += timed
        if rerun_flag:
            return

        logit = reverse_transform(
            paddle.to_tensor(outs[0]), ori_shape, eval_dataset.transforms.transforms, mode="bilinear"
        )
        pred = paddle.to_tensor(logit)
        if len(pred.shape) == 4:  # for humanseg model whose prediction is distribution but not class id
            pred = paddle.argmax(pred, axis=1, keepdim=True, dtype="int32")

        intersect_area, pred_area, label_area = metrics.calculate_area(
            pred, paddle.to_tensor(label), eval_dataset.num_classes, ignore_index=eval_dataset.ignore_index
        )
        intersect_area_all = intersect_area_all + intersect_area
        pred_area_all = pred_area_all + pred_area
        label_area_all = label_area_all + label_area
        if batch_id % 100 == 0:
            print("Eval iter:", batch_id)
            sys.stdout.flush()

        if FLAGS.small_data and batch_id > FLAGS.sample_nums:
            break

    _, miou = metrics.mean_iou(intersect_area_all, pred_area_all, label_area_all)
    _, acc = metrics.accuracy(intersect_area_all, pred_area_all)
    kappa = metrics.kappa(intersect_area_all, pred_area_all, label_area_all)
    _, mdice = metrics.dice(intersect_area_all, pred_area_all, label_area_all)

    time_avg = predict_time / FLAGS.sample_nums
    print(
        "[Benchmark]Batch size: {}, Inference time(ms): min={}, max={}, avg={}".format(
            FLAGS.batch_size, round(time_min * 1000, 2), round(time_max * 1000, 1), round(time_avg * 1000, 1)
        )
    )
    infor = "[Benchmark] #Images: {} mIoU: {:.4f} Acc: {:.4f} Kappa: {:.4f} Dice: {:.4f}".format(
        FLAGS.total_samples, miou, acc, kappa, mdice
    )
    print(infor)
    final_res = {
        "model_name": FLAGS.model_name,
        "jingdu": {
            "value": miou,
            "unit": "mIoU",
        },
        "xingneng": {
            "value": round(time_avg * 1000, 1),
            "unit": "ms",
            "batch_size": FLAGS.batch_size,
        },
    }
    print("[Benchmark][final result]{}".format(final_res))
    sys.stdout.flush()


def main():
    """
    main func
    """
    data_cfg = PaddleSegDataConfig(FLAGS.dataset_config)
    eval_dataset = data_cfg.val_dataset

    batch_sampler = paddle.io.BatchSampler(eval_dataset, batch_size=1, shuffle=False, drop_last=False)
    eval_loader = paddle.io.DataLoader(eval_dataset, batch_sampler=batch_sampler, num_workers=0, return_list=True)
    FLAGS.total_samples = len(eval_dataset) if not FLAGS.small_data else 100
    FLAGS.sample_nums = len(eval_loader) if not FLAGS.small_data else 100
    FLAGS.batch_size = int(FLAGS.total_samples / FLAGS.sample_nums)

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
            use_dynamic_shape=True,
            cpu_threads=FLAGS.cpu_threads,
        )
    elif FLAGS.deploy_backend == "tensorrt":
        model_name = os.path.split(FLAGS.model_path)[-1].rstrip(".onnx")
        engine_file = "{}_{}_model.trt".format(model_name, FLAGS.precision)
        predictor = TensorRTEngine(
            onnx_model_file=FLAGS.model_path,
            shape_info=None,
            max_batch_size=FLAGS.batch_size,
            precision=FLAGS.precision,
            engine_file_path=engine_file,
            calibration_cache_file=FLAGS.calibration_file,
            verbose=False,
        )
    elif FLAGS.deploy_backend == "onnxruntime":
        predictor = ONNXRuntimeEngine(
            onnx_model_file=FLAGS.model_path,
            precision=FLAGS.precision,
            use_trt=FLAGS.use_trt,
            use_mkldnn=FLAGS.use_mkldnn,
            device=FLAGS.device,
        )
    else:
        raise ValueError("deploy_backend not support {}".format(FLAGS.deploy_backend))

    rerun_flag = True if hasattr(predictor, "rerun_flag") and predictor.rerun_flag else False

    eval(predictor, eval_loader, eval_dataset, rerun_flag)

    if rerun_flag:
        print("***** Collect dynamic shape done, Please rerun the program to get correct results. *****")


if __name__ == "__main__":
    parser = argsparser()
    FLAGS = parser.parse_args()

    # DataLoader need run on cpu
    paddle.set_device("cpu")

    main()
