#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python


from dist_tester import Runner

def test_2dp_4pp_mnist():
    """
    2 dp 2pp
    """
    expect = {'gpu0': [2.3073568, 2.3030295], 'gpu1': [2.3073568, 2.3030295], 'gpu2': [2.3073568, 2.3030295], 'gpu3': [2.307356834411621, 2.3030295372009277], 'gpu4': [2.2965198, 2.3025491], 'gpu5': [2.2965198, 2.3025491], 'gpu6': [2.2965198, 2.3025491], 'gpu7': [2.2965197563171387, 2.302549123764038]}
    r = Runner(case_file="dist_SimpleNet_2DP+4PP.py", gpus="0,1,2,3,4,5,6,7", expect=expect)
    r.run()


def test_2dp_2pp_mnist():
    """
    2 dp 2 pp
    """
    expect = {'gpu0': [2.2984905, 2.302197, 2.3058677], 'gpu1': [2.298490524291992, 2.302196979522705, 2.3058676719665527], 'gpu2': [2.2972162, 2.2915306, 2.3032775], 'gpu3': [2.2972161769866943, 2.2915306091308594, 2.3032774925231934]}
    r = Runner(case_file="dist_SimpleNet_2DP+2PP.py", gpus="0,1,2,3", expect=expect)
    r.run()

