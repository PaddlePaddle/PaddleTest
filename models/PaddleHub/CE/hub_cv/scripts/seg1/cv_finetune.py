#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
seg1 finetune
"""
import ast
import argparse
import os
import shutil
import paddle
import numpy as np
import paddlehub as hub
from paddlehub.finetune.trainer import Trainer
from paddlehub.datasets import OpticDiscSeg
from paddlehub.vision.segmentation_transforms import Compose, Resize, Normalize
from paddlehub.vision.utils import ConfusionMatrix


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--model_name", type=str, default=None, help="model name for fine-tuning.")
# parser.add_argument("--version", type=str, default=None, help="model version for fine-tuning.")
parser.add_argument(
    "--use_gpu",
    type=ast.literal_eval,
    default=True,
    help="Whether use GPU for fine-tuning, input should be True or False",
)
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--num_epoch", type=int, default=1, help="Number of epoches for fine-tuning.")
parser.add_argument("--optimizer", type=str, default="AdamW", help="optimizer for training.")
parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate used to train with warmup.")
parser.add_argument("--pretrained", type=str, default=None, help="pretrained model path for training.")
parser.add_argument("--target_size", type=int, default=512, help="target_size of images for training.")
parser.add_argument("--class_num", type=int, default=2, help="target_size of images for training.")
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

    train_transforms = Compose([Resize(target_size=(args.target_size, args.target_size)), Normalize()])
    eval_transforms = Compose([Normalize()])
    train_reader = OpticDiscSeg(train_transforms)
    eval_reader = OpticDiscSeg(eval_transforms, mode="val")

    model = hub.Module(name=args.model_name, num_classes=args.class_num, pretrained=args.pretrained)
    scheduler = paddle.optimizer.lr.PolynomialDecay(
        learning_rate=args.learning_rate, decay_steps=1000, power=0.9, end_lr=0.0001
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

    optimizer = opt(learning_rate=scheduler, parameters=model.parameters())
    trainer = Trainer(model, optimizer, checkpoint_dir=models_save, use_gpu=args.use_gpu)
    trainer.train(train_reader, epochs=args.num_epoch, batch_size=args.batch_size, save_interval=args.save_interval)

    cfm = ConfusionMatrix(eval_reader.num_classes, streaming=True)
    model.eval()
    for imgs, labels in eval_reader:
        imgs = imgs[np.newaxis, :, :, :]
        preds = model(paddle.to_tensor(imgs))[0]
        preds = paddle.argmax(preds, axis=1, keepdim=True).numpy()
        labels = labels[np.newaxis, :, :, :]
        ignores = labels != eval_reader.ignore_index
        cfm.calculate(preds, labels, ignores)
    _, miou = cfm.mean_iou()
    print("miou: {:.4f}".format(miou))
