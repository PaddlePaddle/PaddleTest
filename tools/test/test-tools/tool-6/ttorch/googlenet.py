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

#from util import *


#__init__.py      alexnet.py       googlenet.py     mobilenet.py     quantization/    shufflenetv2.py  vgg.py
#__pycache__/     densenet.py      inception.py     mobilenetv2.py   resnet.py        squeezenet.py    video/
#_utils.py        detection/       mnasnet.py       mobilenetv3.py   segmentation/    utils.py


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.googlenet(pretrained=True).eval()

    #def forward(self, x: torch.Tensor) -> torch.Tensor:
    def forward(self, x):
        with torch.no_grad():
            y_pred = self.model(x)
            return y_pred


def inference():
    # 1.prepare
    device = 'cpu'
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
    #output.to("cpu")
    print(output)


if __name__ == '__main__':
    inference()
