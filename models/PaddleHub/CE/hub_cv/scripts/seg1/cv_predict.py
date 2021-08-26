#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
seg1 predict
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
    "--use_finetune_model",
    type=ast.literal_eval,
    default=False,
    help="Whether use visualizing tool for predict, input should be True or False",
)
parser.add_argument(
    "--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for predict, input should be True or False"
)
parser.add_argument(
    "--visualization",
    type=ast.literal_eval,
    default=True,
    help="Whether use visualizing tool for predict, input should be True or False",
)
parser.add_argument("--checkpoint_dir", type=str, default="./save", help="finetune model for predict.")
parser.add_argument("--output_dir", type=str, default="output_dir", help="results of predict.")
parser.add_argument("--img_path", type=str, default="./../../img_data", help="img for predict.")
args = parser.parse_args()

if __name__ == "__main__":
    pwd = os.getcwd()
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    if args.use_finetune_model is True:
        model = hub.Module(
            name=args.model_name,
            pretrained=os.path.join(args.checkpoint_dir, args.model_name, "epoch_8", "model.pdparams"),
        )
    else:
        model = hub.Module(name=args.model_name, pretrained=None)

    img_list = os.listdir(args.img_path)
    for img in img_list:
        pic = cv2.imread(os.path.join(args.img_path, img))
        model.predict(images=[pic], visualization=args.visualization, save_path=args.output_dir)
    assert len(os.listdir(os.path.join(args.output_dir, "image"))) == len(img_list)
    assert len(os.listdir(os.path.join(args.output_dir, "mask"))) == len(img_list)
