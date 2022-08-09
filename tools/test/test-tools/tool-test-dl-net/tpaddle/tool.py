#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
tpaddle tool.py
"""
import argparse
import paddle
import paddle.vision.models as models


parser = argparse.ArgumentParser(__doc__)
parser.add_argument(
    "--model_name", type=str, default=None, help="model name."
)
args = parser.parse_args()

models_zoo = {'mobilenetv1': models.MobileNetV1,
              'mobilenetv2': models.MobileNetV2,
              'resnet18': models.resnet18,
              'resnet34': models.resnet34,
              'resnet50': models.resnet50,
              'resnet101': models.resnet101,
              'vgg11': models.vgg11,
              'vgg13': models.vgg13,
              'vgg16': models.vgg16,
              'vgg19': models.vgg19}

if __name__ == "__main__":
    model = models_zoo[args.model_name]
    model = model()
    paddle.summary(model, (-1, 3, 224, 224))
