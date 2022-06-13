#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
import argparse
import os
from pathlib import Path
import tensorflow.compat.v1 as tf
from tensorflow.python.client import device_lib


def main(_):
    parser = argparse.ArgumentParser(description='tf_eager_mnist', allow_abbrev=False)
    parser.add_argument('--size', type=int, default=2000)
    parser.add_argument('--dry-run', action='store_true', default=True,
                        help='quickly check a single pass')

    args = parser.parse_args()
    train_size = args.size

    available_divices = device_lib.list_local_devices()
    print('\n'.join(map(str, available_divices)))

    tf.enable_eager_execution()

    mnist_model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
            tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10),
        ]
    )

    (mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32), tf.cast(mnist_labels, tf.int64))
    )
    dataset = dataset.repeat().shuffle(1000).batch(32)


    step_counter = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer()

    for (batch, (images, labels)) in enumerate(dataset.take(train_size)):
        with tf.GradientTape() as tape:
            logits = mnist_model(images, training=True)
            loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)

        grads = tape.gradient(loss_value, mnist_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables), global_step=step_counter)
        if batch % 200 == 0:
            print('Step #%d\tLoss: %.6f' % (batch, loss_value))
        if args.dry_run:
            break


if __name__ == '__main__':
    tf.app.run()
