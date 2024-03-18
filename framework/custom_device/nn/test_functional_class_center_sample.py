#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_class_center_sample
"""

import numpy as np
import paddle
import pytest


class ClassCenterSample(object):
    """
    test_class_center_sample
    """

    def __init__(self, labels, num_classes, num_samples):
        """
        implement
        """
        self.labels = labels
        self.num_classes = num_classes
        self.num_samples = num_samples

    def exception(self, etype):
        """
        exception
        """
        try:
            self.api_result()
        except Exception as e:
            e = str(type(e))
            print(e)
            if etype in e:
                assert True
                print("异常校验成功")
            else:
                assert False, "异常校验失败,异常类型为" + etype

    def api_result(self):
        """
        calculate api result
        """
        if paddle.device.is_compiled_with_cuda():
            paddle.set_device("gpu:0")
        else:
            paddle.set_device("cpu")

        labels = paddle.to_tensor(self.labels, dtype="int32")

        api_remapped_label, api_sampled_class_center = paddle.nn.functional.class_center_sample(
            labels, self.num_classes, self.num_samples
        )
        return api_remapped_label, api_sampled_class_center

    def samples(self):
        """
        calculate positive and negative samples
        """
        positive_samples = list(sorted(set(self.labels)))
        # print(positive_samples)
        negative_samples = list()
        for i in range(10):
            if i not in positive_samples:
                negative_samples.append(i)
        # print(negative_samples)
        return positive_samples, negative_samples

    def remapped_label(self):
        """
        validation remapped_label
        """
        positive_samples = self.samples()[0]
        remapped_label = []
        for i in self.labels:
            remapped_label.append(positive_samples.index(i))

        api_res = self.api_result()
        # print(api_res[0])
        # print(remapped_label)
        # compare
        assert list(api_res[0].numpy()) == remapped_label, "remapped_label验证失败"
        print("remapped_label验证成功")

    def sampled_class_center(self):
        """
        validation sampled_class_center
        """
        positive_samples, negative_samples = self.samples()
        api_res = self.api_result()
        api_sampled_class_center = list(api_res[1].numpy())

        if len(api_sampled_class_center) <= len(positive_samples):
            assert api_sampled_class_center == positive_samples, "sampled_class_center验证失败"
            print("sampled_class_center验证成功")
        else:
            # ① positive samples：
            assert api_sampled_class_center[: len(positive_samples)] == positive_samples, "sampled_class_center验证失败"
            # ② negative samples：
            negative_sampling = api_sampled_class_center[len(positive_samples) :]
            # print(negative_sampling)
            assert len(negative_sampling) == len(set(negative_sampling)), "sampled_class_center验证失败"
            assert len(negative_sampling) == self.num_samples - len(positive_samples), "sampled_class_center验证失败"
            for i in negative_sampling:
                assert i in negative_samples, "sampled_class_center验证失败"
            print("sampled_class_center验证成功")


@pytest.mark.api_nn_class_center_sample_parameters
def test_class_center_sample1():
    """
    default
    """
    labels = list(np.random.randint(0, 10, (5,), dtype="int64"))
    # print(labels)
    num_classes = 10
    num_samples = 8
    t = ClassCenterSample(labels, num_classes, num_samples)
    t.remapped_label()
    t.sampled_class_center()
    print("=" * 50)


@pytest.mark.api_nn_class_center_sample_parameters
def test_class_center_sample2():
    """
    num_samples < len(positive_samples)
    """
    labels = list(np.random.randint(0, 20, (15,), dtype="int64"))
    num_classes = 20
    num_samples = 8
    t = ClassCenterSample(labels, num_classes, num_samples)
    t.remapped_label()
    t.sampled_class_center()
    print("=" * 50)


@pytest.mark.api_nn_class_center_sample_exception
def test_class_center_sample3():
    """
    num_samples > num_classes
    """
    labels = [0, 1, 1, 1]
    num_classes = 5
    num_samples = 8
    t = ClassCenterSample(labels, num_classes, num_samples)
    t.exception("ValueError")
    print("=" * 50)


@pytest.mark.api_nn_class_center_sample_exception
def test_class_center_sample4():
    """
    labels = []
    """
    labels = []
    num_classes = 10
    num_samples = 8
    t = ClassCenterSample(labels, num_classes, num_samples)
    t.exception("ValueError")
    print("=" * 50)
