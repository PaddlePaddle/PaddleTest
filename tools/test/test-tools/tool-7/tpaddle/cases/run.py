#!/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import os
import time
import argparse

case_list = {"avg_pool1D": "test_avg_pool1D",
             "avg_pool2D": "test_avg_pool2D",
             "avg_pool3D": "test_avg_pool3D",
             "conv1d": "test_conv1d",
             "conv2d": "test_conv2d",
             "conv3d": "test_conv3d",
             "Linear": "test_Linear",
             "relu6": "test_relu6",
             "Sigmoid": "test_Sigmoid",
             "tanh": "test_tanh",
             }

desc = case_list.keys()

case_string = "支持OP如下： \n"
for i in desc:
    case_string = case_string + i + "\n"
case_string = case_string + "输入all全部执行。"
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=case_string)
parser.add_argument("op", help="input a op name")
args = parser.parse_args()


def check(res, op):
    if res == 0:
        print("Test {} pass!".format(op))
    else:
        print("Test {} failed! Please check log.".format(op))


def run(op):
    logfile = "test_" + str(int(time.time())) + ".log"
    try:
        if op == "all":
            for v in case_list.values():
                file = v + ".py"
                cmd = "python3.8 -m pytest {}".format(file)
                res = os.system("{} >> {}".format(cmd, logfile))
                check(res, v)
            print("log file is {}".format(logfile))
        elif op in case_list.keys():
            file = case_list[op] + ".py"
            cmd = "python3.8 -m pytest {}".format(file)
            res = os.system("{} >> {}".format(cmd, logfile))
            check(res, op)
            print("log file is {}".format(logfile))
        else:
            raise FileNotFoundError
    except FileNotFoundError as e:
        print("输入有误！")
        print(case_string)


if __name__ == '__main__':
    run(args.op)
