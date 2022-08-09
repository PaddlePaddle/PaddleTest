import tensorflow as tf
import argparse
import numpy as np
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=['save', 'load'])
    parser.add_argument("--content")
    args = parser.parse_args()
    return args

def get_model_kereas():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=[tf.metrics.SparseCategoricalAccuracy()])
    return model

def save_net():
    pwd = sys.path[0]
    model = get_model_kereas()
    model.save("net_tf.pdparams")
    print(' save path : %s/net_tf.pdparams  ' %{pwd})

def load_net():
    pwd = sys.path[0]
    load_main = tf.keras.models.load_model("net_tf.pdparams")
    print(' load path : %s/net_tf.pdparams  ' %{pwd})
    # load_main.summary()

if __name__ == '__main__':
    args = parse_args()
    if args.action == 'save':
            save_net()

    if args.action == 'load':
            load_net()
