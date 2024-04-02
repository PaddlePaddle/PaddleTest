#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python


import numpy as np

def check_all_arrays_equal(lst):
    first_array = lst[0]
    if isinstance(first_array, (np.generic, np.ndarray)):
        for arr in lst[1:]:
            if not np.array_equal(first_array, arr):
                print(first_array)
                print(arr)
                return False
        return True
    elif isinstance(first_array, dict):
        first_array_values = [np.array(x) for x in first_array.values()]
        for arr in lst[1:]:
            arr = [np.array(x) for x in arr.values()]
            if not np.array_equal(first_array_values, arr):
                # print(first_array_values)
                # print(arr)
                return False
        return True
    elif isinstance(first_array, list):
        # grad list
        first_array_values = [x.numpy() for x in first_array]
        for arr in lst[1:]:
            arr = [x.numpy() for x in arr]
            if not np.array_equal(first_array_values, arr):
                # print(first_array_values)
                # print(arr)
                return False
        return True
    else:
        raise TypeError("返回数据类型不能够进行比较")