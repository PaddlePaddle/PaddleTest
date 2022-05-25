#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
darknet53_imagenet
"""

import os
import paddlehub as hub
import paddle

paddle.set_device("gpu")
import cv2


def test_darknet53_imagenet_predict():
    """darknet53_imagenet"""
    os.system("hub install darknet53_imagenet")
    classifier = hub.Module(name="darknet53_imagenet")
    test_img_path = "doc_img.jpeg"
    input_dict = {"image": [test_img_path]}
    result = classifier.classification(data=input_dict)
    print(result)
    os.system("hub uninstall darknet53_imagenet")
