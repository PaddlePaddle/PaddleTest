#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
build module
"""

import argparse
from data.auto_fill_content import auto_filling

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--pr_id", type=str, default=None, help="pr id for ci test")
args = parser.parse_args()


if __name__ == "__main__":
    print(auto_filling("https://github.com/PaddlePaddle/PaddleHub/pull/" + args.pr_id))
