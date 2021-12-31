#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
run win_ce
"""


import os
import sys

pwd = os.getcwd()


def get_files():
    """
    get files
    """

    file_dir = []
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            if name == sys.argv[0]:
                continue
            if os.path.splitext(name)[1] == ".py":
                file_dir.append((root, name))
    return file_dir


bug = 0
bug_list = []
file_dir = get_files()

for dir, file in file_dir:
    os.chdir(dir)
    os.system("echo ============ %s ============" % file)
    if os.system(sys.argv[1] + " " + file):
        bug += 1
        bug_list.append(file)
    os.chdir(pwd)


print("==" * 40)
print("bug: %d" % bug)
for f in bug_list:
    print(f)

if bug:
    sys.exit(1)
