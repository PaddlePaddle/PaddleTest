#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
import re
import sys

regex = [
    r".*@baidu\.com",
    r"(([01]{0,1}\d{0,1}\d|2[0-4]\d|25[0-5])\.){3}([01]{0,1}\d{0,1}\d|2[0-4]\d|25[0-5])",
    r"username",
    r"password",
]


def check(file):
    with open(file) as f:
        for line in f:
            for r in regex:
                if re.search(r, line) is not None:
                    print(file)
                    print(line)
                    exit(1)


if __name__ == "__main__":
    check(sys.argv[1])
