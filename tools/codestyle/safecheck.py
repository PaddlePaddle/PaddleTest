#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
""" code style custom rule """
import re
import sys

regex = [
    r".*@baidu\.com",
    r"(([01]{0,1}\d{0,1}\d|2[0-4]\d|25[0-5])\.){3}([01]{0,1}\d{0,1}\d|2[0-4]\d|25[0-5])",
    r"username",
    r"password",
    r".*@baidu-int\.com",
]

while_list = ["127.0.0.1", "4.4.0.46"]


def check(file):
    """ check """
    try:
        with open(file, encoding="utf-8") as f:
            for line in f:
                for r in regex:
                    match = re.search(r, line)
                    if match and match.group() not in while_list:
                        print("error file:" + file)
                        print("error line:" + line)
                        exit(1)
    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    print("check {}".format(str(sys.argv[1])))
    check(sys.argv[1])
