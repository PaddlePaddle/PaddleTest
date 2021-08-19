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
args = parser.parse_args()

if __name__ == "__main__":
    if args.use_gpu is True:
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")
    embedding = hub.Module(name=args.model_name)
    # 获取单词的embedding
    embedding.search("中国")
    # 计算两个词向量的余弦相似度
    embedding.cosine_sim("中国", "美国")
    # 计算两个词向量的内积
    embedding.dot("中国", "美国")
