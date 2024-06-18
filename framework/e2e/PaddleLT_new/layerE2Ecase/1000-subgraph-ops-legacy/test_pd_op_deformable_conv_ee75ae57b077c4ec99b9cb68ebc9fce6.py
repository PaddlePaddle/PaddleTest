import os
os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_cinn_bucket_compile'] = '1'

import unittest
import numpy as np
import paddle

def GetEnvVarEnableJit():
    enable_jit = os.getenv('PADDLE_DEBUG_ENABLE_JIT')
    return enable_jit not in {
        "0",
        "False",
        "false",
        "OFF",
    }

def GetEnvVarEnableCinn():
    enable_cinn = os.getenv('PADDLE_DEBUG_ENABLE_CINN')
    return enable_cinn not in {
        "0",
        "False",
        "false",
        "OFF",
    }


def GetTolerance(dtype):
    if dtype == np.float16:
        return GetFloat16Tolerance()
    if dtype == np.float32:
        return GetFloat32Tolerance()
    return 1e-6

def GetFloat16Tolerance():
    try:
        return float(os.getenv('PADDLE_DEBUG_FLOAT16_TOL'))
    except:
        return 1e-3

def GetFloat32Tolerance():
    try:
        return float(os.getenv('PADDLE_DEBUG_FLOAT32_TOL'))
    except:
        return 1e-6

def IsInteger(dtype):
    return np.dtype(dtype).char in np.typecodes['AllInteger']


class TestBase:
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def test_train(self):
        dy_outs = self.train(use_cinn=False)
        cinn_outs = self.train(use_cinn=GetEnvVarEnableCinn())

        for cinn_out, dy_out in zip(cinn_outs, dy_outs):
          if type(cinn_out) is list and type(dy_out) is list:
            for x, y in zip(cinn_out, dy_out):
              self.assert_all_close(x, y)
          else:
            self.assert_all_close(cinn_out, dy_out)

    def assert_all_close(self, x, y):
        if (hasattr(x, "numpy") and hasattr(y, "numpy")):
            x_numpy = x.numpy()
            y_numpy = y.numpy()
            assert x_numpy.dtype == y_numpy.dtype
            if IsInteger(x_numpy.dtype):
                np.testing.assert_equal(x_numpy, y_numpy)
            else:
                tol = GetTolerance(x_numpy.dtype)
                np.testing.assert_allclose(x_numpy, y_numpy, atol=tol, rtol=tol)
        else:
            assert x == y



class PrimitiveOp0(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp0(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 128, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 128, 48, 72], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 48, 72], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 48, 72], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp0()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp1(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp1(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 256, 120, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 120, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 120, 200], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 256, 120, 200], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 120, 200], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 120, 200], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp1()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp2(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp2(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 14, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 14, 20], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 256, 14, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 14, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 14, 20], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp2()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp3(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp3(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 256, 48, 72], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 48, 72], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 48, 72], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp3()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp4(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp4(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 258, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 258, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 258, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 16, 16], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp4()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp5(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp5(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 258, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 258, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 258, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 12, 12], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp5()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp6(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp6(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 256, 60, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 60, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 256, 60, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 60, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 60, 100], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp6()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp7(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [0, 0], [1, 1], 1, 1, 1)

class TestPrimitiveOp7(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 128, 120, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2, 120, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 120, 200], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 128, 120, 200], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 120, 200], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 120, 200], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp7()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp8(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp8(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 258, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 258, 24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 258, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 24, 24], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp8()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp9(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp9(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 256, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 256, 36, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 36, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 36, 36], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp9()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp10(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp10(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 258, 15, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 15, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 15, 25], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 258, 15, 25], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 15, 25], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 258, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 15, 25], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp10()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp11(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp11(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 56, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 56, 80], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 256, 56, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 56, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 56, 80], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp11()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp12(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp12(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 256, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 256, 40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 40, 40], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp12()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp13(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp13(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 258, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 258, 40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 258, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 40, 40], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp13()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp14(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp14(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 7, 10], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 256, 7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 7, 10], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp14()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp15(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp15(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 256, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 16, 16], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp15()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp16(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp16(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 512, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 512, 40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 40, 40], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp16()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp17(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp17(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 28, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 28, 40], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 256, 28, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 28, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 28, 40], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp17()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp18(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp18(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 256, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 256, 24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 24, 24], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp18()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp19(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp19(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 128, 60, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 60, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 128, 60, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 60, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 60, 100], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp19()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp20(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp20(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 256, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 12, 12], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp20()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp21(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp21(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 192, 288], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 192, 288], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 256, 192, 288], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 192, 288], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 192, 288], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp21()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp22(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp22(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 256, 30, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 30, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 256, 30, 50], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 30, 50], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 30, 50], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp22()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp23(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp23(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 512, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 512, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 16, 16], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp23()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp24(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp24(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 512, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 512, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 12, 12], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp24()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp25(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp25(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 128, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 128, 96, 144], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 96, 144], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 96, 144], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp25()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp26(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp26(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 128, 30, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 30, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 128, 30, 50], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 30, 50], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 30, 50], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp26()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp27(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp27(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 512, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 512, 36, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 36, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 36, 36], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp27()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp28(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp28(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 258, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 258, 36, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 36, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 258, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 36, 36], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp28()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp29(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp29(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 258, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 24, 36], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 258, 24, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 24, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 258, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 24, 36], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp29()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp30(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp30(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 256, 96, 144], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 96, 144], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 96, 144], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp30()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp31(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [0, 0], [1, 1], 1, 1, 1)

class TestPrimitiveOp31(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 128, 192, 288], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2, 192, 288], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 192, 288], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 128, 192, 288], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 192, 288], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 192, 288], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp31()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp32(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp32(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 112, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 112, 160], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 256, 112, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 112, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 112, 160], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp32()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp33(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

class TestPrimitiveOp33(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 512, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 512, 24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 24, 24], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp33()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

if __name__ == '__main__':
    unittest.main()