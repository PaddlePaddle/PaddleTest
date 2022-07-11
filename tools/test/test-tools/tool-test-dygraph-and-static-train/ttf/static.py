#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
import tensorflow as tf
import math


#导入数据
(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data()
# print(x_train.shape,x_test.shape)


#预处理---正规化
def normalize(x, y):
    x = tf.cast(x, tf.float32)
    x /= 255
    return x, y

#添加一层维度，方便后续扁平化
x_train = tf.expand_dims(x_train,axis=-1)
x_test = tf.expand_dims(x_test,axis=-1)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

#开始定义模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.summary()

# 模型编译
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#开始训练
batch_size = 32
train_dataset = train_dataset.repeat().shuffle(60000).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)
#为tensorboard可视化保存数据
tensorboard_callback = tf.keras.callbacks.TensorBoard(histogram_freq=1)
model.fit(train_dataset, epochs=1, steps_per_epoch=math.ceil(60000/batch_size),
          callbacks=[tensorboard_callback])
