#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
clas2 predict
"""
import os
import shutil
import ast
import argparse
import paddle
import paddlehub as hub
import cv2

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--model_name", type=str, default=None, help="model name for predict.")
parser.add_argument(
    "--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for predict, input should be True or False"
)
parser.add_argument(
    "--visualization",
    type=ast.literal_eval,
    default=True,
    help="Whether use visualization for predict, input should be True or False",
)
parser.add_argument("--output_dir", type=str, default="output_dir", help="img for predict.")
parser.add_argument("--img_path", type=str, default="./../../img_gan2", help="img for predict.")
args = parser.parse_args()

if __name__ == "__main__":
    pwd = os.getcwd()
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    img_list = os.listdir(args.img_path)
    inputs = []
    for img in img_list:
        inputs.append(cv2.imread(os.path.join(args.img_path, img)))

    sr_model = hub.Module(name=args.model_name)
    res = sr_model.reconstruct(
        images=inputs, paths=None, use_gpu=args.use_gpu, visualization=args.visualization, output_dir=args.output_dir
    )
    assert len(os.listdir(args.output_dir)) == 2
