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

import time
import os
import argparse
import yaml
import cv2
import numpy as np

from tqdm import tqdm
import paddle
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig

import paddleocr
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.data import create_operators, transform, build_dataloader
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.metrics import build_metric

logger = get_logger(log_file=__name__)


def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in [".yml", ".yaml"], "only support yaml files for now"
    config = yaml.load(open(file_path, "rb"), Loader=yaml.Loader)
    return config


def get_output_tensors(args, predictor):
    """
    get output tensors func
    """
    output_names = predictor.get_output_names()
    output_tensors = []
    if args.model_type == "rec" and args.rec_algorithm in ["CRNN", "SVTR_LCNet"]:
        output_name = "softmax_0.tmp_0"
        if output_name in output_names:
            return [predictor.get_output_handle(output_name)]
        else:
            for output_name in output_names:
                output_tensor = predictor.get_output_handle(output_name)
                output_tensors.append(output_tensor)
    else:
        for output_name in output_names:
            output_tensor = predictor.get_output_handle(output_name)
            output_tensors.append(output_tensor)
    return output_tensors


def preprocess(image_file, det_limit_side_len, det_limit_type):
    """
    preprocess func
    """
    pre_process_list = [
        {
            "DetResizeForTest": {
                "limit_side_len": det_limit_side_len,
                "limit_type": det_limit_type,
            }
        },
        {
            "NormalizeImage": {
                "std": [0.229, 0.224, 0.225],
                "mean": [0.485, 0.456, 0.406],
                "scale": "1./255.",
                "order": "hwc",
            }
        },
        {"ToCHWImage": None},
        {"KeepKeys": {"keep_keys": ["image", "shape"]}},
    ]
    img = cv2.imread(image_file).astype("float32")
    ori_im = img.copy()
    data = {"image": img}
    data = transform(data, create_operators(pre_process_list))
    return data


def predict_image(predictor, rerun_flag=False):
    """
    predict image func
    """

    # step1: load image and preprocess
    data = preprocess(args.image_file, args.det_limit_side_len, args.det_limit_type)
    img, shape_list = data
    img = np.expand_dims(img, axis=0)
    shape_list = np.expand_dims(shape_list, axis=0)

    predictor.prepare_data([img])

    warmup, repeats = 0, 1
    if args.benchmark:
        warmup, repeats = 20, 100

    for i in range(warmup):
        predictor.run()

    if rerun_flag:
        return

    predict_time = 0.0
    time_min = float("inf")
    time_max = float("-inf")
    for i in range(repeats):
        start_time = time.time()
        predictor.run()
        end_time = time.time()
        timed = end_time - start_time
        time_min = min(time_min, timed)
        time_max = max(time_max, timed)
    predict_time += timed
    monitor.stop()
    time_avg = float(predict_time) / repeats
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
    print(
        "[Benchmark]Inference time(ms): min={}, max={}, avg={}".format(
            round(time_min * 1000, 2), round(time_max * 1000, 1), round(time_avg * 1000, 1)
        )
    )


extra_input_models = ["SRN", "NRTR", "SAR", "SEED", "SVTR", "VisionLAN", "RobustScanner"]


def eval(args):
    """
    eval func
    """
    # DataLoader need run on cpu
    config = load_config(args.dataset_config)
    devices = paddle.set_device("cpu")
    val_loader = build_dataloader(config, "Eval", devices, logger)
    post_process_class = build_post_process(config["PostProcess"])
    eval_class = build_metric(config["Metric"])
    model_type = config["Global"]["model_type"]
    extra_input = True if config["Global"]["algorithm"] in extra_input_models else False

    predictor = load_predictor(args)
    input_names = predictor.get_input_names()
    for name in input_names:
        input_tensor = predictor.get_input_handle(name)
    output_tensors = get_output_tensors(args, predictor)

    with tqdm(
        total=len(val_loader), bar_format="Evaluation stage, Run batch:|{bar}| {n_fmt}/{total_fmt}", ncols=80
    ) as t:
        for batch_id, batch in enumerate(val_loader):
            images = np.array(batch[0])

            input_tensor.copy_from_cpu(images)
            predictor.run()
            outputs = []
            for output_tensor in output_tensors:
                output = output_tensor.copy_to_cpu()
                outputs.append(output)

            batch_numpy = []
            for item in batch:
                batch_numpy.append(np.array(item))

            if args.model_type == "det":
                preds_map = {"maps": outputs[0]}
                post_result = post_process_class(preds_map, batch_numpy[1])
                eval_class(post_result, batch_numpy)
            elif args.model_type == "rec":
                post_result = post_process_class(outputs[0], batch_numpy[1])
                eval_class(post_result, batch_numpy)

            t.update()

    metric = eval_class.get_metric()
    logger.info("metric eval ***************")
    for k, v in metric.items():
        logger.info("{}:{}".format(k, v))


def main():
    """
    main func
    """
    if FLAGS.deploy_backend == "paddle_inference":
        predictor = PaddleInferenceEngine(
            model_dir=FLAGS.model_path,
            precision=FLAGS.precision,
            use_trt=FLAGS.use_trt,
            use_mkldnn=FLAGS.use_mkldnn,
            batch_size=FLAGS.batch_size,
            device=FLAGS.device,
            min_subgraph_size=3,
            use_dynamic_shape=FLAGS.use_dynamic_shape,
            trt_min_shape=1,
            trt_max_shape=1280,
            trt_opt_shape=640,
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

    rerun_flag = True if hasattr(predictor, "rerun_flag") and predictor.rerun_flag else False

    predict_image(predictor, rerun_flag)

    if rerun_flag:
        print("***** Collect dynamic shape done, Please rerun the program to get correct results. *****")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="inference model filepath")
    parser.add_argument("--model_filename", type=str, default="model.pdmodel", help="model file name")
    parser.add_argument("--params_filename", type=str, default="model.pdiparams", help="params file name")
    parser.add_argument("--image_file", type=str, default=None, help="Image path to be processed.")
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
        choices=["fp32", "fp16", "int8"],
        help="The precision of inference. It can be 'fp32', 'fp16' or 'int8'. Default is 'fp16'.",
    )
    parser.add_argument(
        "--deploy_backend",
        type=str,
        default="paddle_inference",
        help="deploy backend, it can be: `paddle_inference`, `tensorrt`, `onnxruntime`",
    )
    parser.add_argument("--use_dynamic_shape", type=bool, default=True, help="Whether use dynamic shape or not.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size of model input.")
    parser.add_argument("--use_mkldnn", type=bool, default=False, help="Whether use mkldnn or not.")
    parser.add_argument("--cpu_threads", type=int, default=1, help="Num of cpu threads.")
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default="max")
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--min_subgraph_size", type=int, default=15)
    parser.add_argument("--model_type", type=str, default="det")
    args = parser.parse_args()
    if args.image_file:
        main()
    else:
        eval(args)
