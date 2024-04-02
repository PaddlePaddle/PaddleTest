#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
compare tool
"""

import os
import argparse
import numpy as np


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--model_name", type=str, default=None, help="model name for compare.")
parser.add_argument("--atol", type=float, default=1e-5, help="largest absolute diff between infer and onnx.")
parser.add_argument("--rtol", type=float, default=1e-5, help="largest relative diff between infer and onnx.")
args = parser.parse_args()

if __name__ == "__main__":
    # model_name = args.model_name
    infer_output_path = os.path.join(args.model_name, "infer_output_np")
    onnx_output_path = os.path.join(args.model_name, "onnx_output_np")
    # infer_output = np.load(infer_output_path)
    # onnx_output = np.load(onnx_output_path)

    infer_list = os.listdir(infer_output_path)
    bug_count = 0
    print("model test formula: np.abs(result - expect) < atol + rtol * np.abs(expect)")
    for i in range(len(infer_list)):
        expect = np.load(os.path.join(infer_output_path, "output_" + str(i) + ".npy"))
        result = np.load(os.path.join(onnx_output_path, "output_" + str(i) + ".npy"))
        res = np.allclose(result, expect, atol=args.atol, rtol=args.rtol, equal_nan=True)
        exp_diff = args.atol + args.rtol * np.abs(expect)
        max_diff = np.max(np.abs(result - expect) - exp_diff)
        diff_num = np.count_nonzero(np.where(np.abs(result - expect) - exp_diff > 0, result - expect, 0))
        pixel_num = result.shape[0] * result.shape[1]
        # 出错打印错误数据
        if result.shape == expect.shape is False:
            print(
                "{} No.{} without_argmax/input cannot pass pixel "
                "label shape comparing test!!!".format(args.model_name, i)
            )
        if res is False:
            print("the {} No.{} without_argmax/input comparing test fail!".format(args.model_name, i))
            # print("the result is {}".format(result))
            # print("the expect is {}".format(expect))
            # print("the exp diff between result and expect is {}".format(exp_diff))
            print("the max diff between result and expect is {}".format(max_diff))
            print("the percent of diff between result and expect is {}".format(diff_num / pixel_num))
            bug_count += 1
        else:
            print("{} No.{} pass prob comparing acc test, nice!!!".format(args.model_name, i))
    print("******" * 30)
    assert bug_count == 0
