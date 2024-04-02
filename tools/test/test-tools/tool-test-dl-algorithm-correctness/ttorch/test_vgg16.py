# !/usr/bin/env python3
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
        self.vgg16 = models.vgg16(pretrained=True).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y_pred = self.vgg16(x)
            return y_pred


def test_gpu_bz1():
    # 1.preprocessed and load model
    device = "cuda"
    transforms = nn.Sequential(
        T.Resize([256, ]),
        T.CenterCrop(224),
        T.ConvertImageDtype(torch.float),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )
    image_tensor = transforms(image_to_tensor("daisy.jpg"))
    input_data = batch_input(image_tensor, batch_size=1)
    predictor = Predictor()
    truth_val = predictor(input_data)

    # 2.to cuda
    input_data = input_data.to(device)
    predictor = predictor.to(device)

    output_val = predictor(input_data)

    check_result(output_val, truth_val)
