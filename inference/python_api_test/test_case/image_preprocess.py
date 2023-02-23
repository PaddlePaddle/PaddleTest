# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
image preprocess
"""
import os
import sys
import time

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


def normalize(num):
    """
    normalize input array elements
    Args:
        num(float): input num
    Returns:
        num(float): return num
    """
    abs_value = abs(num)
    if abs_value > 10:
        length = len(str(int(abs_value)))
        if length >= 3:
            return num / 10 ** (length - 2)
        else:
            return num / 10 ** (length - 1)
    else:
        return num


def sig_fig_compare(array1, array2, delta=5, det_top_bbox=False, need_sort=False, det_top_bbox_threshold=0.75):
    """
    compare significant figure
    Args:
        array1(numpy array): input array 1
        array2(numpy array): input array 2
    Returns:
        diff(numpy array): return diff array
    """
    # start = time.time()
    assert not np.all(np.isnan(array1)), f"output value all nan! \n{array1}"
    if det_top_bbox:
        if len(array1.shape) == 2:
            # 适配部分fp16检测模型case,只对比超过置信度阈值的检测框
            if array1.shape[1] == 6:
                if need_sort:
                    # 将检测框tensor按置信度降序排序
                    array1 = array1[array1[:, 1].argsort()[::-1]]
                    array2 = array2[array2[:, 1].argsort()[::-1]]
                top_count = sum(array1[:, 1] >= det_top_bbox_threshold)
                array1 = array1[:top_count, :]
                array2 = array2[:top_count, :]
        elif len(array1.shape) == 1:
            # 部分检测模型输出检测框数量，在trt fp16下可能与关闭优化的检测框数量不同，跳过，只关注高置信度检测框
            return
    if np.any(abs(array2) > 100):
        normalize_func = np.vectorize(normalize)
        array1_normal = normalize_func(array1)
        array2_normal = normalize_func(array2)
    else:
        array1_normal = array1
        array2_normal = array2
    diff = np.abs(array1_normal - array2_normal)
    diff_count = np.sum(diff > delta)
    print(f"total: {np.size(diff)} diff count:{diff_count} max:{np.max(diff)} delta:{delta}")
    print("output max: ", np.max(abs(array1)), "output min: ", np.min(abs(array1)))
    print("output value debug: ", array1)
    print("output diff array: ", array1[diff > delta])
    print("truth diff array:  ", array2[diff > delta])
    assert diff_count == 0, f"total: {np.size(diff)} diff count:{diff_count} max:{np.max(diff)} delta:{delta}"
    # end = time.time()
    # print(f"精度校验cost：{(end - start) * 1000}ms")
    return diff
