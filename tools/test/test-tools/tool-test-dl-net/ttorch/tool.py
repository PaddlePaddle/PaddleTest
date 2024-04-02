#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
pytorch tool_2.py
"""
import argparse
import torch
import torchvision.models as models
from torchsummary import summary


parser = argparse.ArgumentParser(__doc__)
parser.add_argument(
    "--model_name", type=str, default=None, help="model name."
)
args = parser.parse_args()

models_zoo = {'resnet18': models.resnet18,
              'alexnet': models.alexnet,
              'squeezenet1_0': models.squeezenet1_0,
              'resnet34': models.resnet34,
              'resnet101': models.resnet101,
              'vgg11': models.vgg11,
              'vgg11_bn': models.vgg11_bn,
              'vgg16': models.vgg16}

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(args.model_name + ' is comming!')
    model = models_zoo[args.model_name]
    model = model().to(device)
    summary(model, (3, 224, 224))
