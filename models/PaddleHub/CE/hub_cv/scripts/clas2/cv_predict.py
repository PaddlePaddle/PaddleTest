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
parser.add_argument("--top_k", type=int, default=3, help="top k for predict.")
parser.add_argument("--batch_size", type=int, default=1, help="batch_size for predict.")
parser.add_argument("--img_path", type=str, default=None, help="img for predict.")
args = parser.parse_args()

if __name__ == "__main__":
    pwd = os.getcwd()
    models_save = os.path.join(pwd, "models_save")
    pwd_last = os.path.abspath(os.path.join(os.getcwd(), ".."))
    img_data = os.path.join(pwd_last, "img_data")
    results = os.path.join(pwd, "results")
    if os.path.exists(results):
        shutil.rmtree(results)

    img_list = os.listdir(args.img_path)
    inputs = []
    for img in img_list:
        inputs.append(cv2.imread(os.path.join(args.img_path, img)))

    classifier = hub.Module(name=args.model_name)
    result = classifier.classification(
        images=inputs, batch_size=args.batch_size, use_gpu=args.use_gpu, top_k=args.top_k
    )
    width = classifier.get_expected_image_width()
    height = classifier.get_expected_image_height()
    mean = classifier.get_pretrained_images_mean()
    std = classifier.get_pretrained_images_std()
    print(result)
    print(width)
    print(height)
    print(mean)
    print(std)
