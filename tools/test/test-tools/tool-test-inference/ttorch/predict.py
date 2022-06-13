# !/usr/bin/env python3
import argparse
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torchvision.io import read_image
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    args = parser.parse_args()
    return args



class Predictor(nn.Module):
    def __init__(self):
        super().__init__()

        args = parse_args()
        if args.model== 'alexnet':
            self.model = models.alexnet(pretrained=True).eval()
        elif args.model== 'googlenet':
            self.model = models.googlenet(pretrained=True).eval()
        elif args.model== 'mobilenet':
            self.model = models.mobilenet_v2(pretrained=True).eval()
        elif args.model== 'resnet50':
            self.model = models.wide_resnet50_2(pretrained=True).eval()

    def forward(self, x):
        with torch.no_grad():
            y_pred = self.model(x)
            return y_pred


def inference():
    # 1.prepare
    device = 'cuda'
    transforms = nn.Sequential(
        T.Resize([256, ]),
        T.CenterCrop(224),
        T.ConvertImageDtype(torch.float),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )
    image_tensor = transforms(T.ToTensor()(Image.open("daisy.jpg")))
    batch = torch.stack([image_tensor for _ in range(1)])
    predictor = Predictor()

    # 2.to cuda
    batch = batch.to(device)
    predictor = torch.jit.script(predictor).to(device)

    # 3.infer
    output = predictor(batch)
    print(output)


if __name__ == '__main__':
    inference()
