#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_relu_in_mnist
"""
from __future__ import print_function
import numpy as np
import paddle
from mnist import MNIST, reader_decorator
from custom_relu_mnist import CUSTOMMNIST

paddle.set_device("cpu")


def mnist():
    """
    original mnist
    Returns:

    """
    epoch_num = 1
    BATCH_SIZE = 64

    place = paddle.CPUPlace()

    seed = 33
    np.random.seed(seed)
    paddle.static.default_startup_program().random_seed = seed
    paddle.static.default_main_program().random_seed = seed

    mnist = MNIST()
    adam = paddle.optimizer.Adam(learning_rate=0.001, parameters=mnist.parameters())

    train_reader = paddle.batch(reader_decorator(paddle.dataset.mnist.train()), batch_size=BATCH_SIZE, drop_last=True)

    test_reader = paddle.batch(reader_decorator(paddle.dataset.mnist.test()), batch_size=BATCH_SIZE, drop_last=True)

    train_loader = paddle.io.DataLoader.from_generator(capacity=10)
    train_loader.set_sample_list_generator(train_reader, places=place)

    test_loader = paddle.io.DataLoader.from_generator(capacity=10)
    test_loader.set_sample_list_generator(test_reader, places=place)

    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True

            cost, acc = mnist(img, label)

            loss = paddle.fluid.layers.cross_entropy(cost, label)
            avg_loss = paddle.mean(loss)

            avg_loss.backward()

            adam.minimize(avg_loss)
            # save checkpoint
            mnist.clear_gradients()
            if batch_id % 100 == 0:
                print("Loss at epoch {} step {}: {:}".format(epoch, batch_id, avg_loss.numpy()))
        return avg_loss.numpy()


def custom_mnist():
    """
    custom test
    Returns:

    """
    epoch_num = 1
    BATCH_SIZE = 64

    place = paddle.CPUPlace()

    seed = 33
    np.random.seed(seed)
    paddle.static.default_startup_program().random_seed = seed
    paddle.static.default_main_program().random_seed = seed

    mnist = CUSTOMMNIST()
    adam = paddle.optimizer.Adam(learning_rate=0.001, parameters=mnist.parameters())

    train_reader = paddle.batch(reader_decorator(paddle.dataset.mnist.train()), batch_size=BATCH_SIZE, drop_last=True)

    test_reader = paddle.batch(reader_decorator(paddle.dataset.mnist.test()), batch_size=BATCH_SIZE, drop_last=True)

    train_loader = paddle.io.DataLoader.from_generator(capacity=10)
    train_loader.set_sample_list_generator(train_reader, places=place)

    test_loader = paddle.io.DataLoader.from_generator(capacity=10)
    test_loader.set_sample_list_generator(test_reader, places=place)

    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True

            cost, acc = mnist(img, label)

            loss = paddle.fluid.layers.cross_entropy(cost, label)
            avg_loss = paddle.mean(loss)

            avg_loss.backward()

            adam.minimize(avg_loss)
            # save checkpoint
            mnist.clear_gradients()
            if batch_id % 100 == 0:
                print("Loss at epoch {} step {}: {:}".format(epoch, batch_id, avg_loss.numpy()))
        return avg_loss.numpy()


def test_custom_relu_mnist():
    """
    test custom relu in mnist compare with original mnist
    Returns:

    """
    loss1 = mnist()
    loss2 = custom_mnist()
    assert np.allclose(loss1, loss2, equal_nan=True)
