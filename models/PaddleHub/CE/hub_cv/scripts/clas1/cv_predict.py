#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
clas1 predict
"""
import os
import shutil
import ast
import argparse
import paddle
import paddlehub as hub


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--model_name", type=str, default=None, help="model name for predict.")
parser.add_argument(
    "--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for predict, input should be True or False"
)
parser.add_argument("--img_path", type=str, default=None, help="img for predict.")
args = parser.parse_args()

if __name__ == "__main__":
    pwd = os.getcwd()
    models_save = os.path.join(pwd, "models_save")
    pwd_last = os.path.abspath(os.path.join(os.getcwd(), ".."))
    img_data = os.path.join(pwd_last, "img_data")

    if args.use_gpu is False:
        paddle.set_device("cpu")
    else:
        paddle.set_device("gpu")

    img_list = os.listdir(args.img_path)
    inputs = []
    for i, img in enumerate(img_list):
        inputs.append(os.path.join(args.img_path, img))
    classifier = hub.Module(name=args.model_name)
    input_dict = {"image": inputs}
    result = classifier.classification(data=input_dict)
    print(result)
