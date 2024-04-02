#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
seg2_1 predict
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
parser.add_argument("--batch_size", type=int, default=4, help="batch size for predict.")
parser.add_argument("--output_dir", type=str, default="output_dir", help="results of predict.")
parser.add_argument("--img_path", type=str, default="./../../img_det2", help="img for predict.")
args = parser.parse_args()

if __name__ == "__main__":
    pwd = os.getcwd()
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    model = hub.Module(name=args.model_name)
    img_list = os.listdir(args.img_path)
    inputs_1 = []
    inputs_2 = []
    for img in img_list:
        inputs_1.append(cv2.imread(os.path.join(args.img_path, img)))
        inputs_2.append(os.path.join(args.img_path, img))
    model.segmentation(
        images=inputs_1,
        visualization=args.visualization,
        batch_size=args.batch_size,
        use_gpu=args.use_gpu,
        output_dir=args.output_dir,
    )
    model.segmentation(
        paths=inputs_2,
        visualization=args.visualization,
        batch_size=args.batch_size,
        use_gpu=args.use_gpu,
        output_dir=args.output_dir,
    )
    # assert len(os.listdir(os.path.join(results))) == 3 * len(img_list)
