# !/usr/bin/env python3
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
parser.add_argument("--n_best", type=int, default=3, help="n best sentence.")
parser.add_argument("--beam_size", type=int, default=5, help="beam search width.")
args = parser.parse_args()

if __name__ == "__main__":
    model = hub.Module(name=args.model_name, beam_size=args.beam_size)
    if args.model_name == "transformer_en-de":
        src_texts = [
            "What are you doing now?",
            "The change was for the better; I eat well, I exercise, I take my drugs.",
            "Such experiments are not conducted for ethical reasons.",
        ]
    elif args.model_name == "transformer_zh-en":
        src_texts = ["今天天气怎么样？", "我们一起去吃饭吧。"]
    trg_texts = model.predict(src_texts, n_best=args.n_best, use_gpu=args.use_gpu)
    for idx, st in enumerate(src_texts):
        print("-" * 30)
        print(f"src: {st}")
        for i in range(args.n_best):
            print(f"trg[{i+1}]: {trg_texts[idx*args.n_best+i]}")
