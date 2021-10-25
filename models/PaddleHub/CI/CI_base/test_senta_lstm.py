#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
senta_lstm
"""
import paddlehub as hub

senta = hub.Module(name="senta_lstm")
test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]
results0 = senta.sentiment_classify(texts=test_text, use_gpu=True, batch_size=2)
print(results0)
results1 = senta.get_labels()
print(results1)
results2 = senta.get_vocab_path()
print(results2)
