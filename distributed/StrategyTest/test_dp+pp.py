#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python


from dist_tester import Runner

def test_2dp_4pp_mnist():
    """
    2 dp 2pp
    """
    expect = {'gpu0': [2.308861, 2.303216], 'gpu1': [2.308861, 2.303216], 'gpu2': [2.308861, 2.303216], 'gpu3': [2.308861017227173, 2.303215980529785], 'gpu4': [2.3016574, 2.3023107], 'gpu5': [2.3016574, 2.3023107], 'gpu6': [2.3016574, 2.3023107], 'gpu7': [2.3016574382781982, 2.3023107051849365]}
    r = Runner(case_file="dist_SimpleNet_2DP+4PP.py", gpus="0,1,2,3,4,5,6,7", expect=expect)
    r.run()


def test_2dp_2pp_mnist():
    """
    2 dp 2 pp
    """
    expect = {'gpu0': [2.2919266, 2.3046148, 2.3073053], 'gpu1': [2.291926622390747, 2.304614782333374, 2.307305335998535], 'gpu2': [2.3038232, 2.2936811, 2.304277], 'gpu3': [2.303823232650757, 2.2936811447143555, 2.304276943206787]}
    r = Runner(case_file="dist_SimpleNet_2DP+2PP.py", gpus="0,1,2,3", expect=expect)
    r.run()

