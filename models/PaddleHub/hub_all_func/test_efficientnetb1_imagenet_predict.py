#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
efficientnetb1_imagenet
"""
import os
import paddlehub as hub
import paddle

paddle.set_device("gpu")
import cv2


def test_efficientnetb1_imagenet_predict():
    """efficientnetb1_imagenet"""
    os.system("hub install efficientnetb1_imagenet")
    classifier = hub.Module(name="efficientnetb1_imagenet")
    result = classifier.classification(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = classifier.classification(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall efficientnetb1_imagenet")
