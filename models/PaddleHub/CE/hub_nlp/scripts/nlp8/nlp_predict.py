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
parser.add_argument("--max_seq_len", type=int, default=512, help="max_seq_len for predict.")
parser.add_argument("--batch_size", type=int, default=1, help="batch_size for predict.")
args = parser.parse_args()

if __name__ == "__main__":
    model = hub.Module(name=args.model_name)
    data = [["你是谁？"], ["你好啊。", "吃饭了吗？"]]
    result = model.predict(data, max_seq_len=args.max_seq_len, batch_size=args.batch_size, use_gpu=args.use_gpu)
    print(result)
    assert len(result) == 2
