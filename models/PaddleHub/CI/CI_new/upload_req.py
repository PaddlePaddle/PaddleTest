#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
upload_req
"""
import urllib.request
import argparse


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--pr_id", type=str, default=None, help="pr_id for upload.")
parser.add_argument("--upload_ip", type=str, default=None, help="upload_ip for upload.")
args = parser.parse_args()


if __name__ == "__main__":
    urllib.request.urlopen(args.upload_ip + "/pr/" + args.pr_id)
