#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
emotion_detection_textcnn
"""
import paddlehub as hub

emot = hub.Module(name="emotion_detection_textcnn")
test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]
results0 = emot.emotion_classify(texts=[test_text, test_text], use_gpu=True, batch_size=2)
results1 = emot.get_labels()
results2 = emot.get_vocab_path()
print(results0)
print(results1)
print(results2)
