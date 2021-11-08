#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
finetune audio2
"""
import os
import argparse
import ast
import shutil
import paddle
import paddlehub as hub
from paddlehub.datasets import ESC50


pwd = os.getcwd()
models_save = os.path.join(pwd, "models_save")

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--model_name", type=str, default=None, help="model name for predict.")
parser.add_argument("--task", type=str, default="sound-cls", help="task for predict.")
parser.add_argument(
    "--use_gpu",
    type=ast.literal_eval,
    default=True,
    help="Whether use GPU for fine-tuning, input should be True or False",
)
parser.add_argument("--batch_size", type=int, default=16, help="Total examples' number in batch for training.")
parser.add_argument("--num_epoch", type=int, default=10, help="Number of epoches for fine-tuning.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--save_interval", type=int, default=2, help="Save checkpoint every n epoch.")
parser.add_argument("--checkpoint_dir", type=str, default="./save", help="Directory to model checkpoint")
args = parser.parse_args()

if __name__ == "__main__":
    pwd = os.getcwd()
    if not os.path.exists(os.path.join(pwd, args.checkpoint_dir)):
        os.mkdir(os.path.join(pwd, args.checkpoint_dir))
    models_save = os.path.join(pwd, args.checkpoint_dir, args.model_name)
    if os.path.exists(models_save):
        shutil.rmtree(models_save)
    os.mkdir(os.path.join(pwd, args.checkpoint_dir, args.model_name))

    model = hub.Module(name=args.model_name, task=args.task, num_class=ESC50.num_class)

    train_dataset = ESC50(mode="train")
    dev_dataset = ESC50(mode="dev")

    optimizer = paddle.optimizer.AdamW(learning_rate=args.learning_rate, parameters=model.parameters())

    trainer = hub.Trainer(model, optimizer, checkpoint_dir=models_save, use_gpu=args.use_gpu)
    trainer.train(
        train_dataset,
        epochs=args.num_epoch,
        batch_size=args.batch_size,
        eval_dataset=dev_dataset,
        save_interval=args.save_interval,
    )
