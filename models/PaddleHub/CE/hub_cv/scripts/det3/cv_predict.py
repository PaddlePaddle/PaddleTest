#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
det3 predict
"""
import os
import shutil
import ast
import argparse
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
parser.add_argument("--output_dir", type=str, default="output_dir", help="output dir for predict.")
parser.add_argument("--img_path", type=str, default=None, help="img for predict.")
args = parser.parse_args()

if __name__ == "__main__":
    pwd = os.getcwd()
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    if args.use_gpu is True:
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")

    body_landmark = hub.Module(name=args.model_name)
    img_list = os.listdir(args.img_path)
    result = []
    for img in img_list:
        tmp = body_landmark.predict(
            img=cv2.imread(os.path.join(args.img_path, img)),
            visualization=args.visualization,
            save_path=args.output_dir,
        )
        result.append(tmp)
    print(result)
