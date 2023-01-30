# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
image preprocess
"""
import os
import sys

import cv2
import numpy as np


def read_images_path(images_path, images_size, center=True, model_type="class"):
    """
    read images
    Args:
        images_path(str): images input path
    Returns:
        img_array(numpy): numpy array
    """
    image_names = sorted(os.listdir(images_path))
    images_list = []
    images_origin_list = []
    for name in image_names:
        image_path = os.path.join(images_path, name)
        im = cv2.imread(image_path)
        images_origin_list.append(im)
        images_list.append(preprocess(im, images_size, center, model_type))
    if model_type == "class":
        return images_list
    elif model_type == "det":
        return images_list, images_origin_list


def get_images_npy(npys_path):
    """
    read numpy result
    Args:
        npys_path(str): numpy result path
    Returns:
        result_array(numpy): numpy array
    """
    npy_names = sorted(os.listdir(npys_path))
    npy_list = []
    for name in npy_names:
        npy_path = os.path.join(npys_path, name)
        npy_list.append(np.load(npy_path))
    return npy_list


def read_npy_path(npys_path):
    """
    read numpy result
    Args:
        npys_path(str): numpy result path
    Returns:
        result_array(numpy): numpy array
    """
    npy_names = sorted(os.listdir(npys_path))
    npy_list = []
    for name in npy_names:
        npy_path = os.path.join(npys_path, name)
        npy_list.append(np.load(npy_path))
    return npy_list


def resize_short(img, target_size, model_type="class"):
    """
    resize to target size
    Args:
        img(numpy): img input
        target_size(int): img size
        model_type(str): model type
    Returns:
        resized(numpy): resize img
    """
    if model_type == "class":
        percent = float(target_size) / min(img.shape[0], img.shape[1])
        resized_width = int(round(img.shape[1] * percent))
        resized_height = int(round(img.shape[0] * percent))
        resized = cv2.resize(img, (resized_width, resized_height))
        return resized
    elif model_type == "det":
        im_shape = img.shape
        im_scale_x = float(target_size) / float(im_shape[1])
        im_scale_y = float(target_size) / float(im_shape[0])
        resized = cv2.resize(img, None, None, fx=im_scale_x, fy=im_scale_y)
        return resized


def crop_image(img, target_size, center):
    """
    crop image
    Args:
        img(numpy): img input
        target_size(int): img size
        center(bool): Keep central area or not
    Returns:
        img(numpy): crop image
    """
    height, width = img.shape[:2]
    size = target_size
    if center:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img[int(h_start) : int(h_end), int(w_start) : int(w_end), :]
    return img


def preprocess(img, img_size, center=True, model_type="class"):
    """
    preprocess img
    Args:
        img_size: img size
    Returns:
        img: img add one axis
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = resize_short(img, img_size, model_type)
    img = crop_image(img, img_size, center)
    # bgr-> rgb && hwc->chw
    img = img[:, :, ::-1].astype("float32").transpose((2, 0, 1)) / 255
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std
    return img[np.newaxis, :][0]


def sig_fig_compare(array1, array2, delta=5):
    """
    compare significant figure
    Args:
        num0(float): input num 0
        num1(float): input num 1
    Returns:
        diff(float): return diff
    """
    diff = np.abs(array1 - array2)
    return diff
