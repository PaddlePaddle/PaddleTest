"""
common tools
"""

import numpy as np


def getdtype(dtype="float32"):
    """get dtype"""
    if dtype == "float32" or dtype == "float":
        return np.float32
    if dtype == "float64":
        return np.float64
    if dtype == "int32":
        return np.int32
    if dtype == "int64":
        return np.int64


def dtype2str(dtype=np.float32):
    """dtype2str"""
    if dtype == np.float32:
        return "float32"
    if dtype == np.float64:
        return "float64"
    if dtype == np.int32:
        return "int32"
    if dtype == np.int64:
        return "int64"


def randtool(dtype, low, high, shape):
    """
    np random tools
    """
    data = None
    if dtype.count("int"):
        data = np.random.randint(low, high, shape)
    elif dtype.count("float"):
        data = low + (high - low) * np.random.random(shape)
    elif dtype.count("bool"):
        data = np.random.randint(low, high, shape)
    return data.astype(getdtype(dtype))
