from __future__ import print_function

import tensorflow as tf
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_item")
    args = parser.parse_args()
    return args


def _is_cuda_available():
    gpu_available = tf.test.is_gpu_available()
    res = tf.config.list_physical_devices('GPU')
    assert gpu_available


def _simple_network():
    a = tf.constant(2)
    b = tf.constant(3)
    c = tf.constant(5)

    add = tf.add(a, b)
    res = tf.multiply(add, c)

    print("result =", res.numpy())



if __name__ == '__main__':
    args = parse_args()
    if args.check_item == 'simple_network':
        _simple_network()

    if args.check_item == 'cuda_available':
        _is_cuda_available()
