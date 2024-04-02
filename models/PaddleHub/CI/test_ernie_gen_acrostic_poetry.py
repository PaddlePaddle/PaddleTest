#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
ernie_gen_acrostic_poetry
"""
import paddlehub as hub

module = hub.Module(name="ernie_gen_acrostic_poetry", line=4, word=7)
test_text = ["我喜欢你"]
results0 = module.generate(texts=test_text, use_gpu=True, beam_width=5)
print(results0)
