#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
tensorflow tool_5.py
"""
import os
import argparse
import tensorflow as tf
import tensorflow.keras.applications as applications

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(__doc__)
parser.add_argument(
    "--model_name", type=str, default=None, help="model name."
)
args = parser.parse_args()

models_zoo = {'resnet50': applications.resnet50.ResNet50,
              'vgg16': applications.vgg16.VGG16,
              'mobilenet': applications.mobilenet.MobileNet,
              'efficientnet': applications.efficientnet.EfficientNetB0,
              'resnet101': applications.resnet.ResNet101,
              'densenet121': applications.densenet.DenseNet121}

if __name__ == "__main__":
    img_shape = 224
    model = models_zoo[args.model_name]
    model = model()
    with tf.device("/gpu:0"):
        cifar10 = tf.keras.datasets.cifar10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train[-400:]
        y_train = y_train[-400:]
        x_train = tf.image.resize(x_train, [img_shape, img_shape])
        print('x_train.shape is: ', x_train.shape)
        x_test = x_test[-80:]
        y_test = y_test[-80:]
        x_test = tf.image.resize(x_test, [img_shape, img_shape])
        print('x_test.shape is: ', x_test.shape)
        x_train, x_test = x_train / 255.0, x_test / 255.0

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(x_train, y_train, batch_size=2, epochs=1)

        model.evaluate(x_test, y_test, batch_size=2, verbose=1)
