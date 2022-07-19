#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
generator numpy data tool
"""

import numpy as np
import paddle


def _randtool(dtype, low, high, shape):
    """
    np random tools
    """
    if dtype == "int":
        return np.random.randint(low, high, shape)
    elif dtype == "int32":
        return np.random.randint(low, high, shape).astype("int32")
    elif dtype == "int64":
        return np.random.randint(low, high, shape).astype("int64")
    elif dtype == "float":
        return low + (high - low) * np.random.random(shape)
    elif dtype == "float16":
        return low + (high - low) * np.random.random(shape).astype("float16")
    elif dtype == "float32":
        return low + (high - low) * np.random.random(shape).astype("float32")
    elif dtype == "float64":
        return low + (high - low) * np.random.random(shape).astype("float64")
    elif dtype in ["complex", "complex64", "complex128"]:
        data = low + (high - low) * np.random.random(shape) + (low + (high - low) * np.random.random(shape)) * 1j
        return data if dtype == "complex" or "complex128" else data.astype(np.complex64)
    elif dtype == "bool":
        data = np.random.randint(0, 2, shape).astype("bool")
        return data
    else:
        assert False, "dtype is not supported"
