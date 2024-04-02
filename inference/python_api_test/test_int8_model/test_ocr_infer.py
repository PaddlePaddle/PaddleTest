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
from backend import PaddleInferenceEngine, TensorRTEngine, Monitor

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


def preprocess_det(image_file, det_limit_side_len, det_limit_type):
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


def resize_norm_img_svtr(image_file, image_shape=[3, 48, 320]):
    """
    preprocess func
    """
    img = cv2.imread(image_file).astype("float32")
    imgC, imgH, imgW = image_shape
    resized_image = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
    resized_image = resized_image.astype("float32")
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    return resized_image


def reader_wrapper(reader, input_field="image"):
    """
    reader wrapper func
    """

    def gen():
        for data in reader:
            yield np.array(data[0]).astype(np.float32)

    return gen


def predict_image(predictor, rerun_flag=False):
    """
    predict image func
    """

    # step1: load image and preprocess
    if args.model_type == "det":
        data = preprocess_det(args.image_file, args.det_limit_side_len, args.det_limit_type)
        img, shape_list = data
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
    else:
        img = resize_norm_img_svtr(args.image_file)
        img = np.expand_dims(img, axis=0)

    predictor.prepare_data([img])

    warmup, repeats = 0, 1
    if args.benchmark:
        warmup, repeats = 20, 100

    for i in range(warmup):
        predictor.run()

    if rerun_flag:
        return

    monitor = Monitor(0)
    monitor.start()
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
    time_avg = float(predict_time) / (1.0 * repeats)
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
            round(time_min * 1000, 2), round(time_max * 1000, 2), round(time_avg * 1000, 2)
        )
    )
    final_res = {
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "jingdu": {
            "value": 0,
            "unit": "",
        },
        "xingneng": {
            "value": round(time_avg * 1000, 2),
            "unit": "ms",
        },
        "cpu_mem": {
            "value": cpu_mem,
            "unit": "MB",
        },
        "gpu_mem": {
            "value": gpu_mem,
            "unit": "MB",
        },
    }
    print("[Benchmark][final result]{}".format(final_res))
    sys.stdout.flush()


# eval is not correct
def eval(args, predictor, rerun_flag=False):
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
    repeats = len(val_loader)

    monitor = Monitor(0)
    if not rerun_flag:
        monitor.start()
    predict_time = 0.0
    time_min = float("inf")
    time_max = float("-inf")

    with tqdm(
        total=len(val_loader), bar_format="Evaluation stage, Run batch:|{bar}| {n_fmt}/{total_fmt}", ncols=80
    ) as t:
        for batch_id, batch in enumerate(val_loader):
            images = np.array(batch[0])

            predictor.prepare_data([images])
            start_time = time.time()
            outputs = predictor.run()
            end_time = time.time()
            timed = end_time - start_time
            time_min = min(time_min, timed)
            time_max = max(time_max, timed)
            predict_time += timed

            if rerun_flag:
                return

            batch_numpy = []
            for item in batch:
                batch_numpy.append(np.array(item))

            if model_type == "det":
                preds_map = {"maps": outputs[0]}
                post_result = post_process_class(preds_map, batch_numpy[1])
                eval_class(post_result, batch_numpy)
            elif model_type == "rec":
                post_result = post_process_class(outputs[0], batch_numpy[1])
                eval_class(post_result, batch_numpy)

            t.update()
    print("main pid:", os.getpid())
    monitor.stop()
    print("finish")
    time_avg = float(predict_time) / (1.0 * repeats)
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
            round(time_min * 1000, 2), round(time_max * 1000, 2), round(time_avg * 1000, 2)
        )
    )

    metric = eval_class.get_metric()
    logger.info("metric eval ***************")
    for k, v in metric.items():
        logger.info("{}:{}".format(k, v))

    final_res = {
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "jingdu": {
            "value": 0,
            "unit": "",
        },
        "xingneng": {
            "value": round(time_avg * 1000, 2),
            "unit": "ms",
        },
        "cpu_mem": {
            "value": cpu_mem,
            "unit": "MB",
        },
        "gpu_mem": {
            "value": gpu_mem,
            "unit": "MB",
        },
    }
    print("[Benchmark][final result]{}".format(final_res))
    sys.stdout.flush()


def main(args):
    """
    main func
    """

    val_loader = None
    if args.image_file:
        if args.model_type == "det":
            data = preprocess_det(args.image_file, args.det_limit_side_len, args.det_limit_type)
            img, shape_list = data
        else:
            img = resize_norm_img_svtr(args.image_file)
        img = np.expand_dims(img, axis=0)
        val_loader = [[img]]
    else:
        # DataLoader need run on cpu
        config = load_config(args.dataset_config)
        devices = paddle.set_device("cpu")
        val_loader = build_dataloader(config, "Eval", devices, logger)

    predictor = None
    if args.deploy_backend == "paddle_inference":
        predictor = PaddleInferenceEngine(
            model_dir=args.model_path,
            model_filename=args.model_filename,
            params_filename=args.params_filename,
            precision=args.precision,
            use_trt=args.use_trt,
            use_mkldnn=args.use_mkldnn,
            batch_size=args.batch_size,
            device=args.device,
            min_subgraph_size=3,
            use_dynamic_shape=args.use_dynamic_shape,
            cpu_threads=args.cpu_threads,
        )
    elif args.deploy_backend == "tensorrt":
        model_name = os.path.join(args.model_path, args.model_filename)
        print(model_name)
        engine_file = "{}_{}.trt".format(args.precision, args.batch_size)
        predictor = TensorRTEngine(
            onnx_model_file=model_name,
            shape_info={
                "x": [[1, 3, 100, 100], [1, 3, 800, 800], [1, 3, 1600, 1600]],
            },
            max_batch_size=args.batch_size,
            precision=args.precision,
            engine_file_path=engine_file,
            calibration_cache_file=args.calibration_file,
            calibration_loader=reader_wrapper(val_loader),
            verbose=False,
        )
    if predictor is None:
        return
    rerun_flag = True if hasattr(predictor, "rerun_flag") and predictor.rerun_flag else False

    if args.image_file:
        predict_image(predictor, rerun_flag)
    else:
        eval(args, predictor, rerun_flag)

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
        choices=["paddle_inference", "tensorrt"],
        help="deploy backend, it can be: `paddle`, `tensorrt`, `onnxruntime`",
    )
    parser.add_argument("--calibration_file", type=str, default="calibration.cache")
    parser.add_argument("--use_dynamic_shape", type=bool, default=True, help="Whether use dynamic shape or not.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size of model input.")
    parser.add_argument("--use_mkldnn", type=bool, default=False, help="Whether use mkldnn or not.")
    parser.add_argument("--cpu_threads", type=int, default=1, help="Num of cpu threads.")
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default="max")
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--min_subgraph_size", type=int, default=15)
    parser.add_argument("--model_type", type=str, default="det")
    parser.add_argument("--model_name", type=str, default="", help="model name for benchmark")
    args = parser.parse_args()
    main(args)
