#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
resnet50_vd_imagenet_ssld
"""
import os
import shutil
import paddle
import paddlehub as hub
import paddlehub.vision.transforms as T
from paddlehub.datasets import Flowers
from paddlehub.finetune.trainer import Trainer

pwd = os.getcwd()
models_save = os.path.join(pwd, "models_save")
pwd_last = os.path.abspath(os.path.join(os.getcwd(), ".."))
img_data = os.path.join(pwd_last, "img_data")
results = os.path.join(pwd, "results")
if os.path.exists(results):
    shutil.rmtree(results)

transforms = T.Compose(
    [T.Resize((256, 256)), T.CenterCrop(224), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
    to_rgb=True,
)
flowers = Flowers(transforms)
flowers_validate = Flowers(transforms, mode="val")
model = hub.Module(
    name="resnet50_vd_imagenet_ssld",
    label_list=["roses", "tulips", "daisy", "sunflowers", "dandelion"],
    load_checkpoint=None,
)

optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
trainer = Trainer(model, optimizer, checkpoint_dir=models_save, use_vdl=True, use_gpu=True)

trainer.train(flowers, epochs=2, batch_size=16, eval_dataset=flowers_validate, save_interval=1)
