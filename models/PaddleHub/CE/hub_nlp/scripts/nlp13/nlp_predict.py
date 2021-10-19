#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
nlp predict
"""
import ast
import argparse
import paddlehub as hub


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--model_name", type=str, default=None, help="model name for predict.")
parser.add_argument(
    "--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for predict, input should be True or False"
)
parser.add_argument("--max_length", type=int, default=256, help="max input length.")
parser.add_argument("--max_out_len", type=int, default=256, help="max output length.")
args = parser.parse_args()

if __name__ == "__main__":
    model = hub.Module(name=args.model_name)
    model.__init__(max_length=args.max_length, max_out_len=args.max_out_len)
    # 待预测数据（模拟同声传译实时输入）
    text = [
        "他",
        "他还",
        "他还说",
        "他还说现在",
        "他还说现在正在",
        "他还说现在正在为",
        "他还说现在正在为这",
        "他还说现在正在为这一",
        "他还说现在正在为这一会议",
        "他还说现在正在为这一会议作出",
        "他还说现在正在为这一会议作出安排",
        "他还说现在正在为这一会议作出安排。",
    ]
    for t in text:
        print("input: {}".format(t))
        result = model.translate(t, use_gpu=args.use_gpu)
        print("model output: {}\n".format(result))
