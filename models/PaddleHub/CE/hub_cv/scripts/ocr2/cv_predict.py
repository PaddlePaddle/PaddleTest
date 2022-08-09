#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
ocr2 predict
"""
import os
import shutil
import ast
import argparse
import numpy as np
import cv2
import paddle
import paddlehub as hub


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--model_name", type=str, default=None, help="model name for predict.")
parser.add_argument(
    "--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for predict, input should be True or False"
)
parser.add_argument(
    "--visualization",
    type=ast.literal_eval,
    default=True,
    help="Whether use visualizing tool for predict, input should be True or False",
)
parser.add_argument("--output_dir", type=str, default="output_dir", help="results of predict.")
parser.add_argument("--img_path", type=str, default="./../../img_ocr2", help="img for predict.")
args = parser.parse_args()

if __name__ == "__main__":
    pwd = os.getcwd()
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    ocr = hub.Module(name=args.model_name, use_gpu=args.use_gpu)
    img_list = os.listdir(args.img_path)
    inputs_1 = []
    for img in img_list:
        inputs_1.append(cv2.imread(os.path.join(args.img_path, img)))
    result = ocr.recognize_text(images=inputs_1, visualization=args.visualization, output_dir=args.output_dir)
    for i in range(len(result[0]["data"])):
        print(result[0]["data"][i]["text"])
    assert len(result[0]["data"]) > 20
    # assert len(result[1]['data']) > 0
    assert len(os.listdir(args.output_dir)) == len(inputs_1)
