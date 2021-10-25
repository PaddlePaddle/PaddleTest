# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
nlp finetune
"""
import ast
import argparse
import paddlehub as hub


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--model_name", type=str, default=None, help="model name for predict.")
parser.add_argument(
    "--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for predict, input should be True or False"
)
parser.add_argument("--beam_width", type=int, default=5, help="beam_width for predict.")
args = parser.parse_args()

if __name__ == "__main__":
    module = hub.Module(name=args.model_name)
    test_texts = ["万花丛中过", "片叶不沾身"]
    # generate包含3个参数，texts为输入文本列表，use_gpu指定是否使用gpu，beam_width指定beam search宽度。
    results = module.generate(texts=test_texts, use_gpu=args.use_gpu, beam_width=args.beam_width)
    for result in results:
        print(result)
