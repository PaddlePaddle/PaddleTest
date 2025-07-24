#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
logger init
"""

from .logger import Logger

base_logger = Logger(loggername="FDSentry", save_level="both", log_path="./logs").get_logger()
base_logger.setLevel("INFO")
