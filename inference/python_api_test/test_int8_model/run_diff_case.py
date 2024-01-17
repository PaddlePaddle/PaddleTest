"""
#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import logging

import os
import time
import multiprocessing
import subprocess
import signal
import sys

import argparse
import numpy as np
import yaml

# import cpuinfo

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")


def parse_args():
    """
    parse_args func
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--diff_case_file", type=str, default="./diff_case.txt")
    parser.add_argument("--rerun_turns", type=int, default=3)
    return parser.parse_args()


quant_model_cases = {
    "EfficientNetB0": {
        "test_py": "test_image_classification_infer.py",
        "configs": [("--model_path", "models/EfficientNetB0_QAT")],
    },
    "MobileNetV3_large": {
        "test_py": "test_image_classification_infer.py",
        "configs": [("--model_path", "models/MobileNetV3_large_x1_0_QAT")],
    },
    "PPHGNet_tiny": {
        "test_py": "test_image_classification_infer.py",
        "configs": [("--model_path", "models/PPHGNet_tiny_QAT")],
    },
    "PPLCNetV2": {
        "test_py": "test_image_classification_infer.py",
        "configs": [("--model_path", "models/PPLCNetV2_base_QAT")],
    },
    "ResNet_vd": {
        "test_py": "test_image_classification_infer.py",
        "configs": [("--model_path", "models/ResNet50_vd_QAT")],
    },
    "PPYOLOE": {
        "test_py": "test_ppyoloe_infer.py",
        "configs": [
            ("--model_path", "models/ppyoloe_crn_l_300e_coco_quant"),
            ("--reader_config", "configs/ppyoloe_reader.yml"),
        ],
    },
    "PPYOLOE_PLUS": {
        "test_py": "test_ppyoloe_infer.py",
        "configs": [
            ("--model_path", "models/ppyoloe_plus_crn_s_80e_coco_no_nms_quant"),
            ("--reader_config", "configs/ppyoloe_plus_reader.yml"),
            ("--exclude_nms",),
        ],
    },
    "PicoDet": {
        "test_py": "test_ppyoloe_infer.py",
        "configs": [
            ("--model_path", "models/picodet_s_416_coco_npu_quant"),
            ("--reader_config", "configs/picodet_reader.yml"),
        ],
    },
    "PicoDet_no_postprocess": {
        "test_py": "test_ppyoloe_infer.py",
        "configs": [
            ("--model_path", "models/picodet_s_416_coco_npu_no_postprocess_quant"),
            ("--reader_config", "configs/picodet_reader.yml"),
            ("--exclude_nms",),
        ],
    },
    "YOLOv5s": {"test_py": "test_yolo_series_infer.py", "configs": [("--model_path", "models/yolov5s_quant")]},
    "YOLOv6s": {"test_py": "test_yolo_series_infer.py", "configs": [("--model_path", "models/yolov6s_quant")]},
    "YOLOv7": {"test_py": "test_yolo_series_infer.py", "configs": [("--model_path", "models/yolov7_quant")]},
    "PP-MiniLM": {
        "test_py": "test_nlp_infer.py",
        "configs": [("--model_path", "models/save_ppminilm_afqmc_new_calib"), ("--task_name", "afqmc")],
    },
    "BERT_Base": {
        "test_py": "test_bert_infer.py",
        "configs": [("--model_path", "models/x2paddle_cola_new_calib"), ("--batch_size", 1)],
    },
    "Deeplabv3-ResNet50": {
        "test_py": "test_segmentation_infer.py",
        "configs": [
            ("--model_path", "models/deeplabv3_qat"),
            ("--dataset", "cityscape"),
            ("--dataset_config", "configs/cityscapes_1024x512_scale1.0.yml"),
        ],
    },
    "HRNet": {
        "test_py": "test_segmentation_infer.py",
        "configs": [
            ("--model_path", "models/hrnet_qat"),
            ("--dataset", "cityscape"),
            ("--dataset_config", "configs/cityscapes_1024x512_scale1.0.yml"),
        ],
    },
    "PP-HumanSeg-Lite": {
        "test_py": "test_segmentation_infer.py",
        "configs": [
            ("--model_path", "models/pp_humanseg_qat"),
            ("--dataset", "human"),
            ("--dataset_config", "configs/humanseg_dataset.yaml"),
        ],
    },
    "PP-Liteseg": {
        "test_py": "test_segmentation_infer.py",
        "configs": [
            ("--model_path", "models/pp_liteseg_qat"),
            ("--dataset", "cityscape"),
            ("--dataset_config", "configs/cityscapes_1024x512_scale1.0.yml"),
        ],
    },
    "UNet": {
        "test_py": "test_segmentation_infer.py",
        "configs": [
            ("--model_path", "models/unet_qat"),
            ("--dataset", "cityscape"),
            ("--dataset_config", "configs/cityscapes_1024x512_scale1.0.yml"),
        ],
    },
}


def run_single_case_cmd(model_name=None, l3_cache=False, precision="int8"):
    """
    run diff case func
    """
    if model_name in quant_model_cases:
        cmd = f"python {quant_model_cases[model_name]['test_py']} "
        for conf in quant_model_cases[model_name]["configs"]:
            # print(conf)
            if len(conf) == 2:
                cmd += f"{conf[0]}={conf[1]} "
            elif len(conf) == 1:
                cmd += f"{conf[0]} "
        cmd += f"--precision={precision} --model_name={model_name} --device=XPU "
        if l3_cache:
            cmd += " --use_l3=True"
        return cmd
    else:
        return


def run_single_case(cmd=None, rerun_data=None):
    """
    generate single case cmd func
    """
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)
    print("Output:", result.stdout)
    print("Return Code:", result.returncode)
    benchmark_result = result.stdout.split("======benchmark result======")[-1]
    if benchmark_result:
        benchmark_result = eval(benchmark_result)
        print(benchmark_result)
        rerun_data.append(benchmark_result)
    else:
        print("benchmark error!!!")


def run_diff_case(args):
    """
    main func
    """
    with open(args.diff_case_file, "r", encoding="utf-8") as f:
        data = f.readlines()
    if not data:
        print("no need to rerun!")
        return
    case_list = [i.strip() for i in data]

    for line in case_list:
        tmp = line.split(" ")
        model_name = tmp[0]
        l3_cache = eval(tmp[1])
        precision = tmp[2]
        avg_cost_last = float(tmp[3])
        run_cmd = run_single_case_cmd(model_name, l3_cache, precision)
        if not run_cmd:
            print("diff case error!!!")
            continue

        rerun_data = multiprocessing.Manager().list([])
        best_data = {"avg_cost": float("inf")}
        avg_cost_list = []
        # p90_cost_list = []
        for i in range(args.rerun_turns):
            print(f"rerun_count: {i + 1}")
            print(f"model_name: {model_name}")
            print(f"l3_cache: {l3_cache}")
            print(f"precision: {precision}")
            p = multiprocessing.Process(target=run_single_case, args=(run_cmd, rerun_data))
            p.start()
            p.join()
            p.kill()
            # print("rerun_data: ", rerun_data)
            result_tmp = rerun_data[i]
            try:
                avg_cost_diff = (avg_cost_last - result_tmp["avg_cost"]) / avg_cost_last * 100
            except Exception as e:
                print(e)
                break
            print(f"rerun_{i + 1}_diff:", avg_cost_diff)
            if result_tmp["avg_cost"] < best_data["avg_cost"]:
                best_data = result_tmp
            if abs(avg_cost_diff) < 5:
                best_data = result_tmp
                break
            try:
                avg_cost_list.append(result_tmp["avg_cost"])
            except Exception as e:
                print(e)
                break
        if len(avg_cost_list) == args.rerun_turns:
            avg_array = np.sort(np.array(avg_cost_list))[1 : (args.rerun_turns - 1)]
            best_data["avg_cost"] = float(format(np.mean(avg_array), ".4f"))
            print(avg_cost_list)
        print("best_data: ", best_data)
    # print(case_list)


if __name__ == "__main__":
    args = parse_args()
    run_diff_case(args)
