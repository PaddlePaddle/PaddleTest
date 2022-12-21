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
import sys
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import pkg_resources as pkg

import paddle
from backend import PaddleInferenceEngine, TensorRTEngine, ONNXRuntimeEngine, Monitor
from utils.dataset import COCOValDataset
from utils.yolo_series_post_process import YOLOPostProcess, coco_metric


def argsparser():
    """
    argsparser func
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=str, help="inference model filepath")
    parser.add_argument("--dataset_dir", type=str, default="dataset/coco_val2017", help="COCO dataset dir.")
    parser.add_argument("--val_image_dir", type=str, default="val2017", help="COCO dataset val image dir.")
    parser.add_argument(
        "--val_anno_path", type=str, default="annotations/instances_val2017.json", help="COCO dataset anno path."
    )
    parser.add_argument(
        "--deploy_backend",
        type=str,
        default="paddle_inference",
        help="deploy backend, it can be: `paddle_inference`, `tensorrt`, `onnxruntime`",
    )
    parser.add_argument("--use_dynamic_shape", type=bool, default=True, help="Whether use dynamic shape or not.")
    parser.add_argument("--use_trt", type=bool, default=False, help="Whether use TensorRT or not.")
    parser.add_argument("--precision", type=str, default="paddle", help="mode of running(fp32/fp16/int8)")
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        help="Choose the device you want to run, it can be: CPU/GPU/XPU, default is GPU",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size of model input.")
    parser.add_argument("--use_mkldnn", type=bool, default=False, help="Whether use mkldnn or not.")
    parser.add_argument("--cpu_threads", type=int, default=1, help="Num of cpu threads.")
    parser.add_argument("--calibration_file", type=str, default=None, help="quant onnx model calibration cache file.")
    parser.add_argument("--model_name", type=str, default="", help="model name for benchmark")
    parser.add_argument("--small_data", action="store_true", default=False, help="Whether use small data to eval.")
    return parser


def get_current_memory_mb():
    """
    It is used to Obtain the memory usage of the CPU and GPU during the running of the program.
    And this function Current program is time-consuming.
    """
    import pynvml
    import psutil
    import GPUtil

    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))

    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    cpu_mem = info.uss / 1024.0 / 1024.0
    gpu_mem = 0
    gpu_percent = 0
    gpus = GPUtil.getGPUs()
    if gpu_id is not None and len(gpus) > 0:
        gpu_percent = gpus[gpu_id].load
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem = meminfo.used / 1024.0 / 1024.0
    return round(cpu_mem, 4), round(gpu_mem, 4)


def reader_wrapper(reader, input_field="image"):
    """
    reader wrapper func
    """

    def gen():
        for data in reader:
            yield np.array(data[input_field]).astype(np.float32)

    return gen


def eval(predictor, val_loader, anno_file, rerun_flag=False):
    """
    eval main func
    """
    bboxes_list, bbox_nums_list, image_id_list = [], [], []
    cpu_mems, gpu_mems = 0, 0
    sample_nums = len(val_loader)
    predict_time = 0.0
    time_min = float("inf")
    time_max = float("-inf")
    warmup = 20
    repeats = 20 if FLAGS.small_data else 1
    monitor = Monitor(0)
    if not rerun_flag:
        monitor.start()
    for batch_id, data in enumerate(val_loader):
        data_all = {k: np.array(v) for k, v in data.items()}

        predictor.prepare_data([data_all["image"]])

        for i in range(warmup):
            predictor.run()
            warmup = 0

        start_time = time.time()
        for j in range(repeats):
            outs = predictor.run()
        end_time = time.time()

        timed = (end_time - start_time) / repeats
        time_min = min(time_min, timed)
        time_max = max(time_max, timed)
        predict_time += timed
        if rerun_flag:
            return
        postprocess = YOLOPostProcess(score_threshold=0.001, nms_threshold=0.65, multi_label=True)
        res = postprocess(np.array(outs), data_all["scale_factor"])
        bboxes_list.append(res["bbox"])
        bbox_nums_list.append(res["bbox_num"])
        image_id_list.append(np.array(data_all["im_id"]))
        if batch_id % 100 == 0:
            print("Eval iter:", batch_id)
            sys.stdout.flush()
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
    time_avg = predict_time / sample_nums
    print(
        "[Benchmark]Inference time(ms): min={}, max={}, avg={}".format(
            round(time_min * 1000, 2), round(time_max * 1000, 1), round(time_avg * 1000, 1)
        )
    )

    map_res = coco_metric(anno_file, bboxes_list, bbox_nums_list, image_id_list)
    print("[Benchmark] COCO mAP: {}".format(map_res[0]))
    final_res = {
        "model_name": FLAGS.model_name,
        "jingdu": {
            "value": map_res[0],
            "unit": "mAP",
        },
        "xingneng": {
            "value": round(time_avg * 1000, 1),
            "unit": "ms",
            "batch_size": FLAGS.batch_size,
        },
        "cpu_mem": {
            "value": cpu_mems / sample_nums,
            "unit": "MB",
        },
        "gpu_mem": {
            "value": gpu_mems / sample_nums,
            "unit": "MB",
        },
    }
    print("[Benchmark][final result]{}".format(final_res))
    sys.stdout.flush()


def main():
    """
    main func
    """
    dataset = COCOValDataset(
        dataset_dir=FLAGS.dataset_dir, image_dir=FLAGS.val_image_dir, anno_path=FLAGS.val_anno_path
    )
    anno_file = dataset.ann_file
    val_loader = paddle.io.DataLoader(dataset, batch_size=FLAGS.batch_size, drop_last=True)

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
            calibration_loader=reader_wrapper(val_loader),
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

    eval(predictor, val_loader, anno_file, rerun_flag=rerun_flag)

    if rerun_flag:
        print("***** Collect dynamic shape done, Please rerun the program to get correct results. *****")


if __name__ == "__main__":
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()

    if FLAGS.small_data:
        # set small dataset
        FLAGS.dataset_dir = "dataset/coco"

    # DataLoader need run on cpu
    paddle.set_device("cpu")

    main()
