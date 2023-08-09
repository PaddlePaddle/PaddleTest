#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
# ======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
# ======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file test_dist_fleet_dygraph_loss.py
  * @author liyang109@baidu.com
  * @date 2021-02-03 18:49
  * @brief
  *
  **************************************************************************/
"""
from __future__ import print_function
import subprocess
import pytest


def test_dist_fleet_dygraph_loss_consistent_fleetrun():
    """test_dist_fleet_dygraph_loss_consistent_fleetrun"""
    p = subprocess.Popen(
        "fleetrun --devices=0,1  --job_id dy_loss_fleetrun dist_fleet_dygraph_loss.py",
        shell=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    out, err = p.communicate()
    assert str(out).find("Error") == -1
    assert str(err).find("Error") == -1


def test_dist_fleet_dygraph_loss_consistent_launch():
    """test_dist_fleet_dygraph_loss_consistent_launch"""
    p = subprocess.Popen(
        "python -m paddle.distributed.launch --devices=0,1 --job_id dy_loss_launch \
                    dist_fleet_dygraph_loss.py",
        shell=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    out, err = p.communicate()
    assert str(out).find("Error") == -1
    assert str(err).find("Error") == -1


def test_dist_fleet_dygraph_spawn():
    """test_dist_fleet_dygraph_spawn."""
    p = subprocess.Popen(
        "python dist_fleet_dygraph_spawn.py", shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE
    )
    out, err = p.communicate()
    assert str(out).find("Error") == -1
    assert str(err).find("Error") == -1
