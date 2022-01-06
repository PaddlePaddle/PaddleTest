#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
onnx predict
"""

import os
import argparse
import ast
import numpy as np
import paddle
import onnxruntime as rt


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--model_name", type=str, default=None, help="model name for predict.")
parser.add_argument(
    "--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for predict, input should be True or False"
)
args = parser.parse_args()

if __name__ == "__main__":
    if args.use_gpu is False:
        paddle.set_device("cpu")
    else:
        paddle.set_device("gpu")

    model_path = os.path.join(args.model_name, "onnx_model", "model.onnx")
    input_path = os.path.join(args.model_name, "input_np")
    output_path = os.path.join(args.model_name, "onnx_output_np")

    input_list = os.listdir(input_path)

    for i in range(int(len(input_list) / 3)):
        image = np.load(os.path.join(input_path, "input_" + str(i) + ".npy"))
        im_shape = np.load(os.path.join(input_path, "im_shape_" + str(i) + ".npy"))
        scale_factor = np.load(os.path.join(input_path, "scale_factor_" + str(i) + ".npy"))
        inputs_dict = {}
        sess = rt.InferenceSession(model_path)
        inputs_dict[sess.get_inputs()[0].name] = im_shape
        inputs_dict[sess.get_inputs()[1].name] = image
        inputs_dict[sess.get_inputs()[2].name] = scale_factor
        print(inputs_dict)
        onnx_result = sess.run(None, input_feed=inputs_dict)
        print("{} No.{} input onnx_result:".format(args.model_name, i))
        print(onnx_result[0])
        print(onnx_result[0].shape)
        print("*****" * 30)
        np.save(os.path.join(output_path, "output_" + str(i) + ".npy"), onnx_result[0])
