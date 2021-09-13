#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
nlp predict
"""
import ast
import argparse
import os
import paddlehub as hub


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--model_name", type=str, default=None, help="model name for fine-tuning.")
parser.add_argument("--task", type=str, default="seq-cls", help="model task for fine-tuning.")
parser.add_argument(
    "--use_gpu",
    type=ast.literal_eval,
    default=True,
    help="Whether use GPU for fine-tuning, input should be True or False",
)
parser.add_argument("--max_seq_len", type=int, default=128, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=1, help="Total examples' number in batch for training.")
parser.add_argument(
    "--use_finetune_model", type=ast.literal_eval, default=True, help="Total examples' number in batch for training."
)
args = parser.parse_args()

if __name__ == "__main__":
    pwd = os.getcwd()
    models_save = os.path.join(pwd, "save")

    data = [
        ["这个表情叫什么", "这个猫的表情叫什么"],
        ["什么是智能手环", "智能手环有什么用"],
        ["介绍几本好看的都市异能小说，要完结的！", "求一本好看点的都市异能小说，要完结的"],
        ["一只蜜蜂落在日历上（打一成语）", "一只蜜蜂停在日历上（猜一成语）"],
        ["一盒香烟不拆开能存放多久？", "一条没拆封的香烟能存放多久。"],
    ]
    label_map = {0: "similar", 1: "dissimilar"}
    if args.use_finetune_model is True:
        model = hub.Module(
            name=args.model_name,
            task=args.task,
            load_checkpoint=os.path.join(models_save, args.model_name, "best_model", "model.pdparams"),
            label_map=label_map,
        )
    else:
        model = hub.Module(name=args.model_name, task=args.task, label_map=label_map)
    results = model.predict(data, max_seq_len=args.max_seq_len, batch_size=args.batch_size, use_gpu=args.use_gpu)
    for idx, texts in enumerate(data):
        print("TextA: {}\tTextB: {}\t Label: {}".format(texts[0], texts[1], results[idx]))
    assert len(results) == 5
    for res in results:
        if (res != "similar") and (res != "dissimilar"):
            raise Exception("res does not belong to similar or dissimilar, BUG!!!")
