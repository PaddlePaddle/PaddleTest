"""
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import os
import math
import random
import functools
import numpy as np
import paddle
from PIL import Image, ImageEnhance
from paddle.io import Dataset

random.seed(0)
np.random.seed(0)

DATA_DIM = 224
RESIZE_DIM = 256

THREAD = 16
BUF_SIZE = 10240

DATA_DIR = "data/ILSVRC2012/"
DATA_DIR = os.path.join(os.path.split(os.path.realpath(__file__))[0], DATA_DIR)

img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


def _resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.LANCZOS)
    return img


def _crop_image(img, target_size, center):
    width, height = img.size
    size = target_size
    if center:
        w_start = (width - size) // 2
        h_start = (height - size) // 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


def process_image(sample, mode, color_jitter, rotate, crop_size, resize_size):
    """
    image precess func
    """
    img_path = sample[0]

    try:
        img = Image.open(img_path)
    except:
        print(img_path, "not exists!")
        return None
    img = _resize_short(img, target_size=resize_size)
    img = _crop_image(img, target_size=crop_size, center=True)

    if img.mode != "RGB":
        img = img.convert("RGB")

    img = np.array(img).astype("float32").transpose((2, 0, 1)) / 255
    img -= img_mean
    img /= img_std

    if mode == "train" or mode == "val":
        return img, sample[1]
    elif mode == "test":
        return [img]


class ImageNetDataset(Dataset):
    """
    ImageNet dataset class
    """

    def __init__(self, data_dir=DATA_DIR, mode="val", crop_size=DATA_DIM, resize_size=RESIZE_DIM):
        super(ImageNetDataset, self).__init__()
        self.data_dir = data_dir
        self.crop_size = crop_size
        self.resize_size = resize_size
        val_file_list = os.path.join(data_dir, "val_list.txt")
        self.mode = mode
        with open(val_file_list) as flist:
            lines = [line.strip() for line in flist]
            self.data = [line.split() for line in lines]

    def __getitem__(self, index):
        sample = self.data[index]
        data_path = os.path.join(self.data_dir, sample[0])
        if self.mode == "val":
            data, label = process_image(
                [data_path, sample[1]],
                mode="val",
                color_jitter=False,
                rotate=False,
                crop_size=self.crop_size,
                resize_size=self.resize_size,
            )
            return data, np.array([label]).astype("int64")
        elif self.mode == "test":
            data = process_image(
                [data_path, sample[1]],
                mode="test",
                color_jitter=False,
                rotate=False,
                crop_size=self.crop_size,
                resize_size=self.resize_size,
            )
            return data

    def __len__(self):
        return len(self.data)
