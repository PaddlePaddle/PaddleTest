#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
seg3 predict
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
parser.add_argument("--img_path", type=str, default="./../../img_det2", help="img for predict.")
args = parser.parse_args()

if __name__ == "__main__":
    pwd = os.getcwd()
    model = hub.Module(name=args.model_name)
    img_list = os.listdir(args.img_path)
    for img in img_list:
        inputs_1 = os.path.join(args.img_path, img)
        model.ExtractLine(inputs_1, use_gpu=args.use_gpu)
    # assert len(os.listdir(os.path.join(results))) == 3 * len(img_list)
