#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

from __future__ import print_function

import os
import argparse
from PIL import Image
import numpy
import paddle
import paddle.fluid as fluid


def parse_args():
    parser = argparse.ArgumentParser("mnist")
    parser.add_argument(
        "--enable_ce", action="store_true", help="If set, run the task with continuous evaluation logs."
    )
    parser.add_argument("--use_gpu", type=bool, default=False, help="Whether to use GPU or not.")
    parser.add_argument("--num_epochs", type=int, default=1, help="number of epochs.")
    parser.add_argument("--dry-run", action="store_true", default=True, help="quickly check a single pass")
    args = parser.parse_args()
    return args


def loss_net(hidden, label):
    prediction = fluid.layers.fc(input=hidden, size=10, act="softmax")
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return prediction, avg_loss, acc


def multilayer_perceptron(img, label):
    img = fluid.layers.fc(input=img, size=200, act="tanh")
    hidden = fluid.layers.fc(input=img, size=200, act="tanh")
    return loss_net(hidden, label)


def softmax_regression(img, label):
    return loss_net(img, label)


def convolutional_neural_network(img, label):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img, filter_size=5, num_filters=20, pool_size=2, pool_stride=2, act="relu"
    )
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1, filter_size=5, num_filters=50, pool_size=2, pool_stride=2, act="relu"
    )
    return loss_net(conv_pool_2, label)


def train(nn_type, use_cuda, save_dirname=None, model_filename=None, params_filename=None):
    paddle.enable_static()
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    startup_program = fluid.default_startup_program()
    main_program = fluid.default_main_program()

    if args.enable_ce:
        train_reader = paddle.batch(paddle.dataset.mnist.train(), batch_size=BATCH_SIZE)
        test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)
        startup_program.random_seed = 90
        main_program.random_seed = 90
    else:
        train_reader = paddle.batch(
            paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500), batch_size=BATCH_SIZE
        )
        test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)

    img = paddle.static.data(name="img", shape=[None, 1, 28, 28], dtype="float32")
    label = paddle.static.data(name="label", shape=[None, 1], dtype="int64")

    if nn_type == "softmax_regression":
        net_conf = softmax_regression
    elif nn_type == "multilayer_perceptron":
        net_conf = multilayer_perceptron
    else:
        net_conf = convolutional_neural_network

    prediction, avg_loss, acc = net_conf(img, label)

    test_program = main_program.clone(for_test=True)
    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    def train_test(train_test_program, train_test_feed, train_test_reader):
        acc_set = []
        avg_loss_set = []
        for test_data in train_test_reader():
            acc_np, avg_loss_np = exe.run(
                program=train_test_program, feed=train_test_feed.feed(test_data), fetch_list=[acc, avg_loss]
            )
            acc_set.append(float(acc_np))
            avg_loss_set.append(float(avg_loss_np))
        # get test acc and loss
        acc_val_mean = numpy.array(acc_set).mean()
        avg_loss_val_mean = numpy.array(avg_loss_set).mean()
        return avg_loss_val_mean, acc_val_mean

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)

    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
    exe.run(startup_program)
    epochs = [epoch_id for epoch_id in range(PASS_NUM)]

    lists = []
    step = 0
    for epoch_id in epochs:
        for step_id, data in enumerate(train_reader()):
            metrics = exe.run(main_program, feed=feeder.feed(data), fetch_list=[avg_loss, acc])
            if step % 100 == 0:
                print("Pass %d, Epoch %d, Cost %f" % (step, epoch_id, metrics[0]))
            step += 1
            if args.dry_run:
                break
        # test for epoch
        avg_loss_val, acc_val = train_test(
            train_test_program=test_program, train_test_reader=test_reader, train_test_feed=feeder
        )

        print("Test with Epoch %d, avg_cost: %s, acc: %s" % (epoch_id, avg_loss_val, acc_val))
        lists.append((epoch_id, avg_loss_val, acc_val))
        if save_dirname is not None:
            fluid.io.save_inference_model(
                save_dirname, ["img"], [prediction], exe, model_filename=model_filename, params_filename=params_filename
            )

    # find the best pass
    best = sorted(lists, key=lambda list: float(list[1]))[0]
    print("Best pass is %s, testing Avgcost is %s" % (best[0], best[1]))
    print("The classification accuracy is %.2f%%" % (float(best[2]) * 100))
    paddle.disable_static()


def main(use_cuda, nn_type):
    model_filename = None
    params_filename = None
    save_dirname = "recognize_digits_" + nn_type + ".inference.model"

    # call train() with is_local argument to run distributed train
    train(
        nn_type=nn_type,
        use_cuda=use_cuda,
        save_dirname=save_dirname,
        model_filename=model_filename,
        params_filename=params_filename,
    )


if __name__ == "__main__":
    args = parse_args()
    BATCH_SIZE = 64
    PASS_NUM = args.num_epochs
    use_cuda = args.use_gpu
    # predict = 'softmax_regression' # uncomment for Softmax
    # predict = 'multilayer_perceptron' # uncomment for MLP
    predict = "convolutional_neural_network"  # uncomment for LeNet5
    main(use_cuda=use_cuda, nn_type=predict)
