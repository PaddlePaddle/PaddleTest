#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python


from dist_tester import Runner


def test_2dp_2pp_2tp_mnist():
    """
    2 dp 2 pp 2tp
    """
    expect = {'gpu0': [2.3171508, 2.282558], 'gpu1': [2.3894148, 2.296276], 'gpu2': [2.317150831222534, 2.282557964324951], 'gpu3': [2.3894147872924805, 2.296276092529297], 'gpu4': [2.364337, 2.3121934], 'gpu5': [2.4260015, 2.3250751], 'gpu6': [2.3643369674682617, 2.3121933937072754], 'gpu7': [2.42600154876709, 2.325075149536133]}
    r = Runner(case_file="dist_SimpleNet_2DP+2PP+2TP.py", gpus="0,1,2,3,4,5,6,7", expect=expect)
    r.run()

