#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
nlp finetune
"""
import os
import shutil
import ast
import argparse
import paddle
import paddlehub as hub
from paddlehub.datasets import ChnSentiCorp
from paddlehub.datasets import LCQMC


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--model_name", type=str, default=None, help="model name for fine-tuning.")
# parser.add_argument("--version", type=str, default=None, help="model version for fine-tuning.")
parser.add_argument("--task", type=str, default="seq-cls", help="model task for fine-tuning.")
parser.add_argument("--train_data", type=str, default="dev", help="train data for fine-tuning.")
parser.add_argument("--num_epoch", type=int, default=1, help="Number of epoches for fine-tuning.")
parser.add_argument(
    "--use_gpu",
    type=ast.literal_eval,
    default=True,
    help="Whether use GPU for fine-tuning, input should be True or False",
)
parser.add_argument("--dataset", type=str, default="LCQMC", help="dataset for fine-tuning.")
parser.add_argument("--optimizer", type=str, default="AdamW", help="optimizer for training.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--max_seq_len", type=int, default=128, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--checkpoint_dir", type=str, default="./save", help="Directory to model checkpoint")
parser.add_argument("--save_interval", type=int, default=1, help="Save checkpoint every n epoch.")

args = parser.parse_args()

if __name__ == "__main__":
    pwd = os.getcwd()
    if not os.path.exists(os.path.join(pwd, args.checkpoint_dir)):
        os.mkdir(os.path.join(pwd, args.checkpoint_dir))
    models_save = os.path.join(pwd, args.checkpoint_dir, args.model_name)
    if os.path.exists(models_save):
        shutil.rmtree(models_save)
    os.mkdir(os.path.join(pwd, args.checkpoint_dir, args.model_name))

    model = hub.Module(name=args.model_name, task=args.task)

    if args.dataset == "LCQMC":
        Data = LCQMC
    elif args.dataset == "ChnSentiCorp":
        Data = ChnSentiCorp
    train_dataset = Data(tokenizer=model.get_tokenizer(), max_seq_len=args.max_seq_len, mode="train")
    dev_dataset = Data(tokenizer=model.get_tokenizer(), max_seq_len=args.max_seq_len, mode="dev")
    test_dataset = Data(tokenizer=model.get_tokenizer(), max_seq_len=args.max_seq_len, mode="test")

    if args.optimizer == "AdamW":
        opt = paddle.optimizer.AdamW
    elif args.optimizer == "Adam":
        opt = paddle.optimizer.Adam
    elif args.optimizer == "Adadelta":
        opt = paddle.optimizer.Adadelta
    elif args.optimizer == "Adagrad":
        opt = paddle.optimizer.Adagrad
    elif args.optimizer == "Adamax":
        opt = paddle.optimizer.Adamax
    elif args.optimizer == "Lamb":
        opt = paddle.optimizer.Lamb
    elif args.optimizer == "Momentum":
        opt = paddle.optimizer.Momentum
    elif args.optimizer == "RMSProp":
        opt = paddle.optimizer.RMSProp
    elif args.optimizer == "SGD":
        opt = paddle.optimizer.SGD
    else:
        opt = paddle.optimizer.Optimizer

    optimizer = opt(learning_rate=args.learning_rate, parameters=model.parameters())
    trainer = hub.Trainer(model, optimizer, checkpoint_dir=models_save, use_gpu=args.use_gpu)
    if args.train_data == "train":
        trainer.train(
            train_dataset,
            epochs=args.num_epoch,
            batch_size=args.batch_size,
            eval_dataset=dev_dataset,
            save_interval=args.save_interval,
        )
    elif args.train_data == "test":
        trainer.train(
            test_dataset,
            epochs=args.num_epoch,
            batch_size=args.batch_size,
            eval_dataset=dev_dataset,
            save_interval=args.save_interval,
        )
    else:
        trainer.train(
            dev_dataset,
            epochs=args.num_epoch,
            batch_size=args.batch_size,
            eval_dataset=dev_dataset,
            save_interval=args.save_interval,
        )
    trainer.evaluate(test_dataset, batch_size=args.batch_size)
