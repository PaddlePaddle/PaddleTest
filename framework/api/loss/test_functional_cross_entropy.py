#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_cross_entropy
"""
import paddle
import numpy as np
from paddle.nn.functional import cross_entropy
from apibase import compare
import pytest


def naive_crossentropy(logits, labels, ignore_index, reduction, use_softmax):
    """
    :param logits: data set feature
    :param labels: data set labels
    :param ignore_index
    :param reduction
    """
    loss = []
    data_num, class_num = logits.shape
    count_not_ig = 0  # count not ignore index
    temp = 0
    for class_id in range(0, class_num):
        temp += np.exp(logits[:, class_id])
    for data_id in range(0, data_num):
        if labels[data_id] != ignore_index:
            count_not_ig += 1
        if use_softmax:
            loss_temp = -np.log(np.true_divide(np.exp(logits[:, labels[data_id]]), temp))
        else:
            loss_temp = -np.log(logits[:, labels[data_id]])
        # print("labels is: ", labels)
        # print("labels[data_id] is: ", labels[data_id])
        if labels[data_id] != ignore_index:
            loss.append(loss_temp[data_id])
        else:
            loss.append(0)
    loss = np.array(loss)
    sum_loss = np.array([np.sum(loss)])
    mean_loss = np.array([np.sum(loss) / count_not_ig])
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return mean_loss
    elif reduction == "sum":
        return sum_loss
    else:
        raise Exception("reduction is wrong!!!")


@pytest.mark.api_nn_cross_entropy_parameters
def test_ignore1_reduction_mean_with_softmax():
    """
    test nn.functional.cross_entropy
    ignore_index = 1
    reduction = 'mean'
    use_softmax = True
    """
    np.random.seed(123)
    logits = np.random.rand(4, 8).astype("float32")
    labels = np.array([0, 1, 2, 3]).astype("int64")
    ignore_index = 1
    reduction = "mean"
    pp_loss = cross_entropy(
        paddle.to_tensor(logits), paddle.to_tensor(labels), ignore_index=ignore_index, reduction=reduction
    )
    lzy_loss = naive_crossentropy(logits, labels, ignore_index, reduction, use_softmax=True)
    compare(pp_loss.numpy(), lzy_loss)


@pytest.mark.api_nn_cross_entropy_parameters
def test_ignore1_reduction_mean_with_softmax_float64():
    """
    test nn.functional.cross_entropy
    ignore_index = 1
    reduction = 'mean'
    use_softmax = True
    """
    np.random.seed(123)
    logits = np.random.rand(4, 8).astype("float64")
    labels = np.array([0, 1, 2, 3]).astype("int64")
    ignore_index = 1
    reduction = "mean"
    pp_loss = cross_entropy(
        paddle.to_tensor(logits), paddle.to_tensor(labels), ignore_index=ignore_index, reduction=reduction
    )
    lzy_loss = naive_crossentropy(logits, labels, ignore_index, reduction, use_softmax=True)
    compare(pp_loss.numpy(), lzy_loss)


@pytest.mark.api_nn_cross_entropy_parameters
def test_ignore1_reduction_none_with_softmax():
    """
    test nn.functional.cross_entropy
    ignore_index = 1
    reduction = 'none'
    use_softmax = True
    """
    np.random.seed(123)
    logits = np.random.rand(4, 8).astype("float32")
    labels = np.array([0, 1, 2, 3]).astype("int64")
    ignore_index = 1
    reduction = "none"
    pp_loss = cross_entropy(
        paddle.to_tensor(logits), paddle.to_tensor(labels), ignore_index=ignore_index, reduction=reduction
    )
    lzy_loss = naive_crossentropy(logits, labels, ignore_index, reduction, use_softmax=True)
    compare(pp_loss.numpy(), lzy_loss)


@pytest.mark.api_nn_cross_entropy_parameters
def test_ignore1_reduction_sum_with_softmax():
    """
    test nn.functional.cross_entropy
    ignore_index = 1
    reduction = 'sum'
    use_softmax = True
    """
    np.random.seed(123)
    logits = np.random.rand(4, 8).astype("float32")
    labels = np.array([0, 1, 2, 3]).astype("int64")
    ignore_index = 1
    reduction = "sum"
    pp_loss = cross_entropy(
        paddle.to_tensor(logits), paddle.to_tensor(labels), ignore_index=ignore_index, reduction=reduction
    )
    lzy_loss = naive_crossentropy(logits, labels, ignore_index, reduction, use_softmax=True)
    compare(pp_loss.numpy(), lzy_loss)


@pytest.mark.api_nn_cross_entropy_parameters
def test_ignore2_reduction_none_with_softmax():
    """
    test nn.functional.cross_entropy
    ignore_index = 2
    reduction = 'none'
    use_softmax = True
    """
    np.random.seed(123)
    logits = np.random.rand(4, 8).astype("float32")
    labels = np.array([0, 1, 2, 3]).astype("int64")
    ignore_index = 2
    reduction = "none"
    pp_loss = cross_entropy(
        paddle.to_tensor(logits), paddle.to_tensor(labels), ignore_index=ignore_index, reduction=reduction
    )
    lzy_loss = naive_crossentropy(logits, labels, ignore_index, reduction, use_softmax=True)
    compare(pp_loss.numpy(), lzy_loss)


@pytest.mark.api_nn_cross_entropy_parameters
def test_ignoreno_reduction_none_with_softmax():
    """
    test nn.functional.cross_entropy
    ignore_index = -100
    reduction = 'none'
    use_softmax = True
    """
    np.random.seed(123)
    logits = np.random.rand(4, 8).astype("float32")
    labels = np.array([0, 1, 2, 3]).astype("int64")
    ignore_index = -100
    reduction = "none"
    pp_loss = cross_entropy(
        paddle.to_tensor(logits), paddle.to_tensor(labels), ignore_index=ignore_index, reduction=reduction
    )
    lzy_loss = naive_crossentropy(logits, labels, ignore_index, reduction, use_softmax=True)
    compare(pp_loss.numpy(), lzy_loss)


@pytest.mark.api_nn_cross_entropy_parameters
def test_ignoreno_reduction_none_without_softmax():
    """
    test nn.functional.cross_entropy
    ignore_index = -100
    reduction = 'none'
    use_softmax = False
    """
    np.random.seed(123)
    logits = np.random.rand(4, 8).astype("float32")
    labels = np.array([0, 1, 2, 3]).astype("int64")
    ignore_index = -100
    reduction = "none"
    pp_loss = cross_entropy(
        paddle.to_tensor(logits),
        paddle.to_tensor(labels),
        ignore_index=ignore_index,
        reduction=reduction,
        use_softmax=False,
    )
    lzy_loss = naive_crossentropy(logits, labels, ignore_index, reduction, use_softmax=False)
    compare(pp_loss.numpy(), lzy_loss)


@pytest.mark.api_nn_cross_entropy_parameters
def test_ignore1_reduction_none_without_softmax():
    """
    test nn.functional.cross_entropy
    ignore_index = 1
    reduction = 'none'
    use_softmax = False
    """
    np.random.seed(123)
    logits = np.random.rand(4, 8).astype("float32")
    labels = np.array([0, 1, 2, 3]).astype("int64")
    ignore_index = 1
    reduction = "none"
    pp_loss = cross_entropy(
        paddle.to_tensor(logits),
        paddle.to_tensor(labels),
        ignore_index=ignore_index,
        reduction=reduction,
        use_softmax=False,
    )
    lzy_loss = naive_crossentropy(logits, labels, ignore_index, reduction, use_softmax=False)
    compare(pp_loss.numpy(), lzy_loss)


@pytest.mark.api_nn_cross_entropy_parameters
def test_ignore1_reduction_mean_without_softmax():
    """
    test nn.functional.cross_entropy
    ignore_index = 1
    reduction = 'mean'
    use_softmax = False
    """
    np.random.seed(123)
    logits = np.random.rand(4, 8).astype("float32")
    labels = np.array([0, 1, 2, 3]).astype("int64")
    ignore_index = 1
    reduction = "mean"
    pp_loss = cross_entropy(
        paddle.to_tensor(logits),
        paddle.to_tensor(labels),
        ignore_index=ignore_index,
        reduction=reduction,
        use_softmax=False,
    )
    lzy_loss = naive_crossentropy(logits, labels, ignore_index, reduction, use_softmax=False)
    compare(pp_loss.numpy(), lzy_loss)


@pytest.mark.api_nn_cross_entropy_parameters
def test_softlabe():
    """
    test nn.functional.cross_entropy
    set weight when use softlabel
    """
    w_set = paddle.ones((5,))
    logits = paddle.ones((3, 5)) * 4
    labels = paddle.ones((3, 5))
    res = paddle.nn.functional.cross_entropy(logits, labels, soft_label=True, weight=w_set)
    exp = np.array([8.04718971])
    compare(res.numpy(), exp)


# np.random.seed(123)
# logits = np.random.rand(4, 8).astype("float32")
# labels = np.array([0, 1, 2, 3]).astype("int64")
# ignore_index = 1
# reduction = 'none'
# pp_loss_true = cross_entropy(paddle.to_tensor(logits), paddle.to_tensor(labels),
#                         ignore_index=ignore_index, reduction=reduction, use_softmax=True)
# pp_loss_false = cross_entropy(paddle.to_tensor(logits), paddle.to_tensor(labels),
#                         ignore_index=ignore_index, reduction=reduction, use_softmax=False)
# print(pp_loss_true)
# print("***" * 20)
# print(pp_loss_false)
