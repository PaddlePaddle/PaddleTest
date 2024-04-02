#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
clas3 predict
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
    help="Whether use visualization for predict, input should be True or False",
)
parser.add_argument("--checkpoint_dir", type=str, default="save", help="finetune model for predict.")
parser.add_argument("--save_path", type=str, default="save_path", help="save_path for predict.")
parser.add_argument("--img_path", type=str, default="./../../img_gan4", help="img for predict.")
args = parser.parse_args()

if __name__ == "__main__":
    if args.use_gpu is True:
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")

    if args.use_finetune_model is True:
        model = hub.Module(
            name=args.model_name,
            load_checkpoint=os.path.join(args.checkpoint_dir, args.model_name, "best_model", "model.pdparams"),
        )
    else:
        model = hub.Module(name=args.model_name, load_checkpoint=None)

    model.set_config(prob=0.1)
    img_list = os.listdir(args.img_path)
    for img in img_list:
        pic = os.path.join(args.img_path, img)
        result = model.predict(images=[pic], visualization=args.visualization, save_path=args.save_path)
        print(result)
        assert result[0] != []
