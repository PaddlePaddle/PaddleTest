#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python


from dist_tester import Runner

def test_double_mnist():
    """
    测试双卡MNIST 数据并行策略
    """
    expect = {'gpu0': [2.2984905, 2.302523, 2.2963464, 2.2918725, 2.3048432], 'gpu1': [2.298490524291992, 2.302522897720337, 2.296346426010132, 2.291872501373291, 2.3048431873321533]}
    r = Runner(case_file="dist_SimpleNet_2_Cards_PP.py", gpus="0,1", expect=expect)
    r.run()


def test_quadra_mnist():
    """
    4 cards
    """
    expect = {'gpu0': [2.3073568, 2.2928421, 2.3058], 'gpu1': [2.3073568, 2.2928421, 2.3058], 'gpu2': [2.3073568, 2.2928421, 2.3058], 'gpu3': [2.307356834411621, 2.292842149734497, 2.305799961090088]}
    r = Runner(case_file="dist_SimpleNet_4_Cards_PP.py", gpus="0,1,2,3", expect=expect)
    r.run()


def test_hexa_mnist():
    """
    6 cards
    """
    expect = {'gpu0': [2.30574, 2.2944226], 'gpu1': [2.30574, 2.2944226], 'gpu2': [2.30574, 2.2944226], 'gpu3': [2.30574, 2.2944226], 'gpu4': [2.30574, 2.2944226], 'gpu5': [2.3057401180267334, 2.2944226264953613]}
    r = Runner(case_file="dist_SimpleNet_6_Cards_PP.py", gpus="0,1,2,3,4,5", expect=expect)
    r.run()

def test_octa_mnist():
    """
    8 cards
    """
    expect = {'gpu0': [2.3019652, 2.3021924], 'gpu1': [2.3019652, 2.3021924], 'gpu2': [2.3019652, 2.3021924], 'gpu3': [2.3019652, 2.3021924], 'gpu4': [2.3019652, 2.3021924], 'gpu5': [2.3019652, 2.3021924], 'gpu6': [2.3019652, 2.3021924], 'gpu7': [2.3019652366638184, 2.302192449569702]}
    r = Runner(case_file="dist_SimpleNet_8_Cards_PP.py", gpus="0,1,2,3,4,5,6,7", expect=expect)
    r.run()
