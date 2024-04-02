#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
video1 predict
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
parser.add_argument("--visualization", type=ast.literal_eval, default=True, help="Whether use visualization")
parser.add_argument("--draw_threshold", type=float, default=0.5, help="draw_threshold for predict.")
parser.add_argument("--video_path", type=str, default="./../../video_data", help="video for predict.")
parser.add_argument("--tracking_output_dir", type=str, default="mot_result_gpu", help="tracking output dir.")
parser.add_argument("--stream_output_dir", type=str, default="image_stream_output", help="stream mode output dir.")
args = parser.parse_args()

if __name__ == "__main__":
    pwd = os.getcwd()
    tracker = hub.Module(name=args.model_name)
    video_list = os.listdir(args.video_path)
    for video in video_list:
        inputs = os.path.join(args.video_path, video)
        tracker.tracking(
            inputs, output_dir=args.tracking_output_dir, visualization=True, draw_threshold=0.5, use_gpu=args.use_gpu
        )
        with tracker.stream_mode(
            output_dir=args.stream_output_dir, visualization=True, draw_threshold=0.5, use_gpu=True
        ):
            img_list = os.listdir(os.path.join(args.tracking_output_dir, "mot_outputs", "MOT16-14-raw"))
            inputs_ = []
            for i, img in enumerate(img_list):
                inputs_.append(cv2.imread(os.path.join(args.tracking_output_dir, "mot_outputs", "MOT16-14-raw", img)))
            tracker.predict(inputs_)
