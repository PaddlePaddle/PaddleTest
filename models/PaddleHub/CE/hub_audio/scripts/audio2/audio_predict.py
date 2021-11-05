#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
audio2 predict
"""
import os
import ast
import argparse
import librosa
import paddlehub as hub
from paddlehub.env import MODULE_HOME
from paddlehub.datasets import ESC50


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--model_name", type=str, default=None, help="model name for predict.")
parser.add_argument("--task", type=str, default="sound-cls", help="task for predict.")
parser.add_argument(
    "--use_finetune_model",
    type=ast.literal_eval,
    default=False,
    help="Whether use visualizing tool for predict, input should be True or False",
)
parser.add_argument(
    "--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for predict, input should be True or False"
)
parser.add_argument("--checkpoint_dir", type=str, default="./save", help="auido path for predict.")
parser.add_argument("--audio_path", type=str, default="./../../audio_data", help="auido path for predict.")
args = parser.parse_args()

if __name__ == "__main__":
    pwd = os.getcwd()
    wav = os.path.join(args.audio_data, "LJ001-0003.wav")  # 存储在本地的需要预测的wav文件
    sr = 44100  # 音频文件的采样率
    label_map = {idx: label for idx, label in enumerate(ESC50.label_list)}
    if args.use_finetune_model is True:
        checkpoint = (os.path.join(args.checkpoint_dir, args.model_name, "best_model", "model.pdparams"),)
    else:
        checkpoint = None

    model = hub.Module(
        name=args.model_name, task=args.task, num_class=ESC50.num_class, label_map=label_map, load_checkpoint=checkpoint
    )
    data = [librosa.load(wav, sr=sr)[0]]
    result = model.predict(data, sample_rate=sr, batch_size=1, feat_type="mel", use_gpu=True)
    print(result[0])  # result[0]包含音频文件属于各类别的概率值
    print("***" * 20)
    topk = 10  # 展示音频得分前10的标签和分数
    # 读取audioset数据集的label文件
    label_file = os.path.join(MODULE_HOME, "panns_cnn10", "audioset_labels.txt")
    label_map = {}
    with open(label_file, "r") as f:
        for i, l in enumerate(f.readlines()):
            label_map[i] = l.strip()
    data = [librosa.load(wav, sr=sr)[0]]
    result = model.predict(data, sample_rate=sr, batch_size=1, feat_type="mel", use_gpu=False)
    msg = []
    # 打印topk的类别和对应得分
    for label, score in list(result[0].items())[:topk]:
        msg += f"{label}: {score}\n"
    print(msg)
