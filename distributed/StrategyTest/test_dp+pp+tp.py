#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python


from dist_tester import Runner


def test_2dp_2pp_2tp_mnist():
    """
    2 dp 2 pp 2tp
    """
    expect = {'gpu0': [2.4363656, 2.2911491], 'gpu1': [2.420516, 2.3203418], 'gpu2': [2.4363656044006348, 2.291149139404297], 'gpu3': [2.420516014099121, 2.3203418254852295], 'gpu4': [2.5139077, 2.3063989], 'gpu5': [2.4914849, 2.321696], 'gpu6': [2.5139076709747314, 2.306398868560791], 'gpu7': [2.4914848804473877, 2.3216960430145264]}
    r = Runner(case_file="dist_SimpleNet_2DP+2PP+2TP.py", gpus="0,1,2,3,4,5,6,7", expect=expect)
    r.run()

