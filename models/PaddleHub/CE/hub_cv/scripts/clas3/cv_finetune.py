#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
clas3 finetune
"""
import ast
import argparse
import os
import paddle
import numpy as np
import paddlehub as hub
from paddlehub.finetune.trainer import Trainer
from paddlehub.datasets import OpticDiscSeg
from paddlehub.vision.segmentation_transforms import Compose, Resize, Normalize
from paddlehub.datasets import Flowers
from paddlehub.vision.utils import ConfusionMatrix
import paddlehub.vision.transforms as T
import cv2
import shutil


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--model_name", type=str, default=None, help="model name for fine-tuning.")
# parser.add_argument("--version", type=str, default=None, help="model version for fine-tuning.")
parser.add_argument(
    "--use_gpu",
    type=ast.literal_eval,
    default=True,
    help="Whether use GPU for fine-tuning, input should be True or False",
)
parser.add_argument("--batch_size", type=int, default=16, help="Total examples' number in batch for training.")
parser.add_argument("--num_epoch", type=int, default=1, help="Number of epoches for fine-tuning.")
parser.add_argument("--optimizer", type=str, default="Adam", help="optimizer for training.")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate used to train with warmup.")
parser.add_argument("--target_size", type=int, default=256, help="target_size of images for training.")
parser.add_argument(
    "--use_vdl",
    type=ast.literal_eval,
    default=True,
    help="Whether use use_vdl for fine-tuning, input should be True or False",
)
parser.add_argument("--pretrained", type=str, default="None", help="pretrained model path for training.")
parser.add_argument("--save_interval", type=int, default=1, help="Save checkpoint every n epoch.")
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
    transforms = T.Compose(
        [
            T.Resize((args.target_size, args.target_size)),
            T.CenterCrop(224),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
        to_rgb=True,
    )
    flowers = Flowers(transforms)
    flowers_validate = Flowers(transforms, mode="val")
    if args.pretrained == "None":
        model = hub.Module(
            name=args.model_name,
            label_list=["roses", "tulips", "daisy", "sunflowers", "dandelion"],
            load_checkpoint=None,
        )
    else:
        model = hub.Module(
            name=args.model_name,
            label_list=["roses", "tulips", "daisy", "sunflowers", "dandelion"],
            load_checkpoint=args.pretrained,
        )
    if args.optimizer == "Adam":
        opt = paddle.optimizer.Adam
    elif args.optimizer == "AdamW":
        opt = paddle.optimizer.AdamW
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

    optimizer = paddle.optimizer.Adam(learning_rate=args.learning_rate, parameters=model.parameters())
    trainer = Trainer(model, optimizer, checkpoint_dir=models_save, use_vdl=args.use_vdl, use_gpu=args.use_gpu)
    trainer.train(
        flowers,
        epochs=args.num_epoch,
        batch_size=args.batch_size,
        eval_dataset=flowers_validate,
        save_interval=args.save_interval,
    )
