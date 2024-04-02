# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
nlp predict
"""
import ast
import argparse
import paddle
import paddlehub as hub


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--model_name", type=str, default=None, help="model name for predict.")
parser.add_argument(
    "--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for predict, input should be True or False"
)
parser.add_argument("--batch_size", type=int, default=1, help="batch_size for predict.")
parser.add_argument(
    "--return_tag",
    type=ast.literal_eval,
    default=True,
    help="Whether use tag for predict, input should be True or False",
)
args = parser.parse_args()

if __name__ == "__main__":
    lac = hub.Module(name="lac")
    test_text = ["今天是个好日子", "天气预报说今天要下雨", "下一班地铁马上就要到了"]
    results0 = lac.cut(text=test_text, use_gpu=args.use_gpu, batch_size=args.batch_size, return_tag=args.return_tag)
    print(results0)
