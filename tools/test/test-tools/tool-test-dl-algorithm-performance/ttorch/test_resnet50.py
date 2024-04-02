# !/usr/bin/env python3
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

from util import *


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=True).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y_pred = self.resnet50(x)
            return y_pred


def test(args):
    # 1.preprocessed and load model
    device = "cuda" if args.use_gpu else "cpu"
    transforms = nn.Sequential(
        T.Resize([256, ]),
        T.CenterCrop(224),
        T.ConvertImageDtype(torch.float),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )
    image_tensor = transforms(image_to_tensor("daisy.jpg"))
    predictor = Predictor()

    # 2.to cuda
    input_data = image_tensor.to(device)
    predictor = predictor.to(device)

    # 3.benchmark
    total_time = inference(predictor, input_data, args)
    summary_config(args, total_time)


if __name__ == '__main__':
    args = parse_args()
    test(args)
