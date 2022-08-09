#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test jit dict for
"""
import paddle


def func1():
    """
    func1
    """
    pos = [1, 3]
    neg = [-1, -3]
    dict_val = {"minus": 0}
    # test `zip` with `for`
    for (x, y) in zip(pos, neg):
        val = x - y
        dict_val.update({k: val + dict_val[k] for k, v in dict_val.items()})
    return dict_val


def func2():
    """
    func2
    """
    pos = [1, 3]
    neg = [-1, -3]
    dict_val = {"minus": 0}
    # test `zip` with `for`
    for i, (x, y) in enumerate(zip(pos, neg)):
        val = x - y
        dict_val.update({k: val + dict_val[k] for k, v in dict_val.items()})

    return dict_val


def func3():
    """
    func3
    """
    pos = [1, 3]
    neg = [-1, -3]
    dict_val = {"minus": 0}
    # test `zip` with `for`
    for (x, y) in zip(pos, neg):
        val = x * 2 + y
        dict_val.update({k: val - dict_val[k] for k, v in dict_val.items()})
    return dict_val


def func4():
    """
    func4
    """
    pos = [1, 3, 5, 3, 3, 2, 5, 1]
    neg = [-1, -3, 2, 4, 3, 2, 1, 7]
    dict_val = {"minus": 0}
    # test `zip` with `for`
    for (x, y) in zip(pos, neg):
        val = x * 3 + y
        dict_val.update({k: val - dict_val[k] for k, v in dict_val.items()})
    return dict_val


def func5():
    """
    func5
    """
    pos = [1, 3, 5, 3, 3, 2, 5, 1]
    neg = [-1, -3, 2, 4, 3, 2, 1, 7]
    dict_val = {"minus": 0, "add": 0}
    # test `zip` with `for`
    for (x, y) in zip(pos, neg):
        val1 = x * 3 + y
        val2 = x - y
        dict_val.update({k: val2 - dict_val[k] + val1 for k, v in dict_val.items()})
    return dict_val


def test_func1():
    """
    test func1
    """
    res1 = paddle.jit.to_static(func1)()
    assert res1 == {"minus": 8}


def test_func2():
    """
    test func2
    """
    res2 = paddle.jit.to_static(func2)()
    assert res2 == paddle.jit.to_static(func2)()


def test_func3():
    """
    test func3
    """
    res2 = paddle.jit.to_static(func3)()
    assert res2 == paddle.jit.to_static(func3)()


def test_func4():
    """
    test func4
    """
    res2 = paddle.jit.to_static(func4)()
    assert res2 == paddle.jit.to_static(func4)()


def test_func5():
    """
    test func5
    """
    res2 = paddle.jit.to_static(func5)()
    assert res2 == paddle.jit.to_static(func5)()
