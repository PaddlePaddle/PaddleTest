#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_conj
"""
import os
import unittest
from distutils.sysconfig import get_python_lib
import paddle
import paddle.static as static
from paddle.utils.cpp_extension import load, get_build_directory
from paddle.utils.cpp_extension.extension_utils import run_cmd
from paddle.utils.cpp_extension.extension_utils import IS_WINDOWS
import numpy as np

current_path = os.path.dirname(os.path.abspath(__file__))

site_packages_path = get_python_lib()
paddle_includes = [
    os.path.join(site_packages_path, "paddle", "include"),
    os.path.join(site_packages_path, "paddle", "include", "third_party"),
]

# Test for extra compile args
extra_cc_args = ["-w", "-g"] if not IS_WINDOWS else ["/w"]
extra_nvcc_args = ["-O3"]
extra_compile_args = {"cc": extra_cc_args, "nvcc": extra_nvcc_args}
# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = "{}\\custom_relu_module_jit\\custom_relu_module_jit.pyd".format(get_build_directory())
if os.name == "nt" and os.path.isfile(file):
    cmd = "del {}".format(file)
    run_cmd(cmd, True)

custom_ops = load(
    name="custom_conj_jit",
    sources=[current_path + "/conj.cc"],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_cc_args,  # test for cc flags
    extra_cuda_cflags=extra_nvcc_args,  # test for nvcc flags
    verbose=True,
)


def is_complex(dtype):
    """
    check whether complex number
    Args:
        dtype:

    Returns:

    """
    return dtype == paddle.fluid.core.VarDesc.VarType.COMPLEX64 or dtype == paddle.fluid.core.VarDesc.VarType.COMPLEX128


def to_complex(dtype):
    """
    change to complex number
    Args:
        dtype:

    Returns:

    """
    if dtype == "float32":
        return np.complex64
    elif dtype == "float64":
        return np.complex128
    else:
        return dtype


def conj_dynamic(func, dtype, np_input):
    """
    dynamic train
    Args:
        func:
        dtype:
        np_input:

    Returns:

    """
    paddle.set_device("cpu")
    x = paddle.to_tensor(np_input)
    out = func(x)
    out.stop_gradient = False
    sum_out = paddle.sum(out)
    if is_complex(sum_out.dtype):
        sum_out.real().backward()
    else:
        sum_out.backward()
    return out.numpy(), x.grad


def conj_static(func, shape, dtype, np_input):
    """
    static train
    Args:
        func:
        shape:
        dtype:
        np_input:

    Returns:

    """
    paddle.enable_static()
    paddle.set_device("cpu")
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name="x", shape=shape, dtype=dtype)
            x.stop_gradient = False
            out = func(x)
            sum_out = paddle.sum(out)
            static.append_backward(sum_out)

            exe = static.Executor()
            exe.run(static.default_startup_program())

            out_v, x_grad_v = exe.run(
                static.default_main_program(), feed={"x": np_input}, fetch_list=[out.name, x.name + "@GRAD"]
            )
    paddle.disable_static()
    return out_v, x_grad_v


class TestCustomConjJit(unittest.TestCase):
    """
    test case
    """

    def setUp(self):
        """
        set up
        Returns:

        """
        self.dtypes = ["float32", "float64"]
        self.shape = [2, 20, 2, 3]

    def check_output(self, out, pd_out, name):
        """
        check result
        Args:
            out:
            pd_out:
            name:

        Returns:

        """
        self.assertTrue(
            np.array_equal(out, pd_out), "custom op {}: {},\n paddle api {}: {}".format(name, out, name, pd_out)
        )

    def run_dynamic(self, dtype, np_input):
        """
        run dynamic
        Args:
            dtype:
            np_input:

        Returns:

        """
        out, x_grad = conj_dynamic(custom_ops.custom_conj, dtype, np_input)
        pd_out, pd_x_grad = conj_dynamic(paddle.conj, dtype, np_input)

        self.check_output(out, pd_out, "out")
        self.check_output(x_grad, pd_x_grad, "x's grad")

    def run_static(self, dtype, np_input):
        """
        run static
        Args:
            dtype:
            np_input:

        Returns:

        """
        out, x_grad = conj_static(custom_ops.custom_conj, self.shape, dtype, np_input)
        pd_out, pd_x_grad = conj_static(paddle.conj, self.shape, dtype, np_input)

        self.check_output(out, pd_out, "out")
        self.check_output(x_grad, pd_x_grad, "x's grad")

    def test_dynamic(self):
        """test dynamic"""
        for dtype in self.dtypes:
            np_input = np.random.random(self.shape).astype(dtype)
            self.run_dynamic(dtype, np_input)

    def test_static(self):
        """
        test static
        Returns:

        """
        for dtype in self.dtypes:
            np_input = np.random.random(self.shape).astype(dtype)
            self.run_static(dtype, np_input)

    # complex only used in dynamic mode now
    def test_complex_dynamic(self):
        """
        test complex dynamic
        Returns:

        """
        for dtype in self.dtypes:
            np_input = np.random.random(self.shape).astype(dtype) + 1j * np.random.random(self.shape).astype(dtype)
            self.run_dynamic(to_complex(dtype), np_input)
