#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python


from dist_tester import Runner

def test_2pp_4tp_mnist():
    """
    2 dp 4 tp
    """
    expect = {'gpu0': [2.3033006, 2.307656], 'gpu1': [2.3065155, 2.2905607], 'gpu2': [2.310104, 2.302918], 'gpu3': [2.2932801, 2.3044891], 'gpu4': [2.303300619125366, 2.3076560497283936], 'gpu5': [2.3065154552459717, 2.290560722351074], 'gpu6': [2.3101038932800293, 2.302917957305908], 'gpu7': [2.2932801246643066, 2.3044891357421875]}
    r = Runner(case_file="dist_SimpleNet_4TP+2PP.py", gpus="0,1,2,3,4,5,6,7", expect=expect)
    r.run()


def test_2pp_2tp_mnist():
    """
    2 dp 2 tp
    """
    expect = {'gpu0': [2.304707, 2.3005917, 2.313837], 'gpu1': [2.302135, 2.290606, 2.2990162], 'gpu2': [2.3047070503234863, 2.3005917072296143, 2.3138370513916016], 'gpu3': [2.3021349906921387, 2.2906060218811035, 2.299016237258911]}
    r = Runner(case_file="dist_SimpleNet_2TP+2PP.py", gpus="0,1,2,3", expect=expect)
    r.run()

