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
parser.add_argument("--model_type", type=str, default=None, help="model type for compare.")
parser.add_argument("--atol", type=float, default=1e-5, help="largest absolute diff between infer and onnx.")
parser.add_argument("--rtol", type=float, default=1e-5, help="largest relative diff between infer and onnx.")
args = parser.parse_args()

if __name__ == "__main__":
    # model_name = args.model_name
    infer_output_path = os.path.join(args.model_name, "infer_output_np")
    onnx_output_path = os.path.join(args.model_name, "onnx_output_np")

    bug_count = 0
    # if not os.listdir(infer_output_path):
    #     infer_empty = False
    #     bug_count += 1
    #     print('{} infer_output_path is empty!'.format(args.model_name))
    # else:
    #     infer_empty = True
    #
    # if not os.listdir(onnx_output_path):
    #     onnx_empty = False
    #     bug_count += 1
    #     print('{} onnx_output_path is empty!'.format(args.model_name))
    # else:
    #     onnx_empty = True
    # assert bug_count == 0

    if args.model_type == "det":
        if os.path.exists(os.path.join(infer_output_path, "det_results.txt")):
            infer_empty = True
        else:
            infer_empty = False
            bug_count += 1
            print("{} infer_output_path is empty!".format(args.model_name))
        if os.path.exists(os.path.join(onnx_output_path, "det_results.txt")):
            onnx_empty = True
        else:
            onnx_empty = False
            bug_count += 1
            print("{} onnx_output_path is empty!".format(args.model_name))
        assert bug_count == 0

        with open(os.path.join(infer_output_path, "det_results.txt"), "r") as f:
            infer_data = f.readlines()
        with open(os.path.join(onnx_output_path, "det_results.txt"), "r") as f:
            onnx_data = f.readlines()
        res = infer_data == onnx_data
        if res is False:
            print("det model comparing test fail!")
        else:
            print("det model comparing test success!")

        assert res

    elif args.model_type == "rec":
        if os.path.exists(os.path.join(infer_output_path, "rec_results.txt")):
            infer_empty = True
        else:
            infer_empty = False
            bug_count += 1
            print("{} infer_output_path is empty!".format(args.model_name))
        if os.path.exists(os.path.join(onnx_output_path, "rec_results.txt")):
            onnx_empty = True
        else:
            onnx_empty = False
            bug_count += 1
            print("{} onnx_output_path is empty!".format(args.model_name))
        assert bug_count == 0

        with open(os.path.join(infer_output_path, "rec_results.txt"), "r") as f:
            infer_data = f.readlines()
        with open(os.path.join(onnx_output_path, "rec_results.txt"), "r") as f:
            onnx_data = f.readlines()
        index = 0
        for i, o in zip(infer_data, onnx_data):
            i_list = i.split(", ", 1)
            o_list = o.split(", ", 1)
            res = i_list[0] == o_list[0]
            if res is False:
                bug_count += 1
                print(
                    "rec model No.{} word comparing test fail! infer_result: {}, onnx_result: {}".format(
                        index, i_list[0], o_list[0]
                    )
                )
            else:
                print("rec model No.{} word comparing test success!".format(index))

            infer = np.array(float(i_list[1]))
            onnx = np.array(float(o_list[1]))
            if np.isnan(infer):
                continue
            else:
                res = np.allclose(infer, onnx, atol=args.atol, rtol=args.rtol)
                if res is False:
                    bug_count += 1
                    print(
                        "rec model No.{} prob comparing test fail! infer_result: {}, onnx_result: {}".format(
                            index, infer, onnx
                        )
                    )
                else:
                    print("rec model No.{} prob comparing test success!".format(index))
                index += 1

        assert bug_count == 0
