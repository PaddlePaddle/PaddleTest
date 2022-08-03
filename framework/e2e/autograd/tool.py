"""
tool
"""
import numpy as np


class FrontAPIBase(object):
    """
    convert class to function
    """

    def __init__(self, func):
        """initialize"""
        self.api = func

    def encapsulation(self, *args, **kwargs):
        """class to func"""
        obj = self.api(**kwargs)
        return obj(*args)

    def exe(self, *args, **kwargs):
        """run"""
        return self.encapsulation(*args, **kwargs)


NPDTYPE = {
    "paddle.float16": np.float16,
    "paddle.float32": np.float32,
    "paddle.float64": np.float64,
    "paddle.int32": np.int32,
    "paddle.int64": np.int64,
    "paddle.complex64": np.complex64,
    "paddle.complex128": np.complex128,
    "paddle.bool": np.bool,
}
