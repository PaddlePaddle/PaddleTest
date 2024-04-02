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
parser.add_argument("--batch_size", type=int, default=2, help="batch size for predict.")
args = parser.parse_args()

if __name__ == "__main__":
    simnet_bow = hub.Module(name=args.model_name)
    test_text_1 = ["这道题太难了", "这道题太难了", "这道题太难了"]
    test_text_2 = ["这道题是上一年的考题", "这道题不简单", "这道题很有意思"]
    test_text = [test_text_1, test_text_2]
    results = simnet_bow.similarity(texts=test_text, use_gpu=args.use_gpu, batch_size=args.batch_size)
    print(results)
