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
parser.add_argument("--beam_width", type=int, default=5, help="beam search width.")
args = parser.parse_args()

if __name__ == "__main__":
    module = hub.Module(name=args.model_name)
    test_text = ["我喜欢你"]
    results0 = module.generate(texts=test_text, use_gpu=args.use_gpu, beam_width=args.beam_width)
    print(results0)
