#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
simnet_bow
"""
import paddlehub as hub

simnet_bow = hub.Module(name="simnet_bow")
test_text_1 = ["这道题太难了", "这道题太难了", "这道题太难了"]
test_text_2 = ["这道题是上一年的考题", "这道题不简单", "这道题很有意思"]
test_text = [test_text_1, test_text_2]
results = simnet_bow.similarity(texts=test_text, use_gpu=True, batch_size=2)
print(results)
