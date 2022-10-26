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


def load_predictor(args):
    """
    load predictor func
    """
    model_file = os.path.join(args.model_path, args.model_filename)
    params_file = os.path.join(args.model_path, args.params_filename)
    if not os.path.exists(model_file):
        raise ValueError("{} doesn't exist".format(model_file))
    if not os.path.exists(params_file):
        raise ValueError("{} doesn't exist".format(params_file))
    pred_cfg = PredictConfig(model_file, params_file)
    pred_cfg.enable_memory_optim()
    pred_cfg.switch_ir_optim(True)

    precision_map = {
        "int8": PrecisionType.Int8,
        "fp32": PrecisionType.Float32,
        "fp16": PrecisionType.Half,
    }

    if args.device == "GPU":
        pred_cfg.enable_use_gpu(100, 0)
        if args.use_trt:
            pred_cfg.enable_tensorrt_engine(
                workspace_size=1 << 30,
                precision_mode=precision_map[args.precision],
                max_batch_size=args.max_batch_size,
                min_subgraph_size=args.min_subgraph_size,
                use_calib_mode=False,
            )

            # collect shape
            trt_shape_f = os.path.join(args.model_path, f"trt_dynamic_shape.txt")

            if not os.path.exists(trt_shape_f):
                pred_cfg.collect_shape_range_info(trt_shape_f)
                print(f"collect dynamic shape info into : {trt_shape_f}")
            else:
                print(f"dynamic shape info file( {trt_shape_f} ) already exists, not need to generate again.")

            pred_cfg.enable_tuned_tensorrt_dynamic_shape(trt_shape_f, True)
    else:
        pred_cfg.disable_gpu()
        pred_cfg.set_cpu_math_library_num_threads(args.cpu_threads)
        if args.use_mkldnn:
            pred_cfg.enable_mkldnn()
            if args.precision == "int8":
                pred_cfg.enable_mkldnn_int8({"conv2d", "depthwise_conv2d", "pool2d", "elementwise_mul"})
    # pred_cfg.disable_glog_info()
    pred_cfg.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
    pred_cfg.delete_pass("matmul_transpose_reshape_fuse_pass")
    pred_cfg.switch_use_feed_fetch_ops(False)

    predictor = create_predictor(pred_cfg)
    return predictor


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


def predict_image(args):
    """
    predict image func
    """

    # step1: load image and preprocess
    data = preprocess(args.image_file, args.det_limit_side_len, args.det_limit_type)
    img, shape_list = data
    img = np.expand_dims(img, axis=0)
    shape_list = np.expand_dims(shape_list, axis=0)

    # Step2: Prepare prdictor
    predictor = load_predictor(args)

    # Step3: Inference
    input_names = predictor.get_input_names()
    for name in input_names:
        input_tensor = predictor.get_input_handle(name)
    output_tensors = get_output_tensors(args, predictor)

    input_tensor.copy_from_cpu(img)

    warmup, repeats = 0, 1
    if args.benchmark:
        warmup, repeats = 20, 100

    for i in range(warmup):
        predictor.run()

    start_time = time.time()
    for i in range(repeats):
        predictor.run()
        outputs = []
        for output_tensor in output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
    total_time = time.time() - start_time
    avg_time = float(total_time) / repeats
    print(f"[Benchmark]Average inference time: \033[91m{round(avg_time*1000, 2)}ms\033[0m")
    """
    if args.model_type == 'det':
        preds_map = {'maps': outputs[0]}
        post_result = post_process_class(preds_map, shape_list)
    elif args.model_type == 'rec':
        post_result = post_process_class(outputs[0], shape_list)
    """


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
    parser.add_argument("--use_mkldnn", type=bool, default=False, help="Whether use mkldnn or not.")
    parser.add_argument("--cpu_threads", type=int, default=1, help="Num of cpu threads.")
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default="max")
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--min_subgraph_size", type=int, default=15)
    parser.add_argument("--model_type", type=str, default="det")
    args = parser.parse_args()
    if args.image_file:
        predict_image(args)
    else:
        eval(args)
