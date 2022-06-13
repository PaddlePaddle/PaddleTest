#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
ttf tool_2.py
"""
import argparse
import tensorflow.keras.applications as applications


parser = argparse.ArgumentParser(__doc__)
parser.add_argument(
    "--model_name", type=str, default=None, help="model name."
)
args = parser.parse_args()

models_zoo = {'resnet50': applications.resnet50.ResNet50,
              'vgg16': applications.vgg16.VGG16,
              'mobilenet': applications.mobilenet.MobileNet,
              'inception_v3': applications.inception_v3.InceptionV3,
              'efficientnet': applications.efficientnet.EfficientNetB0,
              'resnet101': applications.resnet.ResNet101,
              'xception': applications.xception.Xception,
              'densenet121': applications.densenet.DenseNet121}

if __name__ == "__main__":
    model = models_zoo[args.model_name]
    model = model(
        include_top=True, weights=None, input_tensor=None,
        input_shape=None, pooling=None, classes=1000
    )
    model.summary()
