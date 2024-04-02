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
parser.add_argument("--max_steps", type=int, default=300, help="max_steps for predict.")
parser.add_argument("--batch_size", type=int, default=2, help="batch_size for predict.")
parser.add_argument("--module_name", type=str, default="test_ernie_gen_module", help="module name for export.")
parser.add_argument("--author", type=str, default="test", help="author name for diy module.")
args = parser.parse_args()

if __name__ == "__main__":
    module = hub.Module(name=args.model_name)
    result = module.finetune(
        train_path="train.txt",
        dev_path="dev.txt",
        use_gpu=args.use_gpu,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )
    module.export(params_path=result["last_save_path"], module_name=args.module_name, author=args.author)
