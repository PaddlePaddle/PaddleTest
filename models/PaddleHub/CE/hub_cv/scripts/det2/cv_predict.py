#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
det2 predict
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
parser.add_argument("--batch_size", type=int, default=2, help="batch size for predict.")
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
    inputs1 = []
    inputs2 = []
    for img in img_list:
        inputs1.append(cv2.imread(os.path.join(args.img_path, img)))
        inputs2.append(os.path.join(args.img_path, img))
    pose_landmark = hub.Module(name=args.model_name)
    result = pose_landmark.keypoint_detection(
        images=inputs1,
        batch_size=args.batch_size,
        use_gpu=args.use_gpu,
        visualization=args.visualization,
        output_dir=args.output_dir,
    )
    result1 = pose_landmark.keypoint_detection(
        paths=inputs2,
        batch_size=args.batch_size,
        use_gpu=args.use_gpu,
        visualization=args.visualization,
        output_dir=args.output_dir,
    )
    print(result)
    print(result1)
    # assert len(os.listdir(os.path.join(args.output_dir))) == 4
    points_list = [
        "left_ankle",
        "left_knee",
        "left_hip",
        "right_hip",
        "right_knee",
        "right_ankle",
        "pelvis",
        "thorax",
        "upper_neck",
        "head_top",
        "right_wrist",
        "right_elbow",
        "right_shoulder",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
    ]
    for i in [0, 1]:
        for res, res1 in zip(result[i]["data"].keys(), result1[i]["data"].keys()):
            assert res == res1
    for i in [0, 1]:
        for pt in points_list:
            assert np.array_equal(np.array(result[i]["data"][pt]), np.array(result1[i]["data"][pt]))
