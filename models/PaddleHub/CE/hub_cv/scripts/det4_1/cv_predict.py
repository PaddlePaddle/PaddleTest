#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
det4 predict
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
parser.add_argument("--score_thresh", type=float, default=0.15, help="score_thresh for predict.")
parser.add_argument(
    "--visualization",
    type=ast.literal_eval,
    default=True,
    help="Whether use visualizing tool for predict, input should be True or False",
)
parser.add_argument("--output_dir", type=str, default="output_dir", help="img for predict.")
parser.add_argument("--img_path", type=str, default=None, help="img for predict.")
args = parser.parse_args()

if __name__ == "__main__":
    pwd = os.getcwd()
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    img_list = os.listdir(args.img_path)
    face_detector = hub.Module(name=args.model_name)
    inputs1 = []
    inputs2 = []
    for img in img_list:
        inputs1.append(cv2.imread(os.path.join(args.img_path, img)))
        inputs2.append(os.path.join(args.img_path, img))
    result = face_detector.face_detection(
        images=inputs1,
        score_thresh=args.score_thresh,
        use_gpu=args.use_gpu,
        visualization=args.visualization,
        output_dir=args.output_dir,
    )
    result1 = face_detector.face_detection(
        paths=inputs2,
        score_thresh=args.score_thresh,
        use_gpu=args.use_gpu,
        visualization=args.visualization,
        output_dir=args.output_dir,
    )
    print(result)
    print(result1)
