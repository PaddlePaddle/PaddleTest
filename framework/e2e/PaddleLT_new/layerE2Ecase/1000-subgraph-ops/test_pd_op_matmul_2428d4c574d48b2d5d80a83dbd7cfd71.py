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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp0(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 1174, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 1174, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp1(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 65536, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 65536, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp2(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([22, 197, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[22, 197, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp3(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([22, 197, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[22, 197, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp4(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 100, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 384], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp5(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 1, 1, 7, 7, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 1, 1, 7, 7, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp6(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 512, 4096], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4096, 512], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 512, 4096], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4096, 512], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp7(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 4096], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 4096], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp8(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 100, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 384], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp9(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 156], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([156, 39], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 156], dtype='float32'),
            paddle.static.InputSpec(shape=[156, 39], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp10(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 39], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([39, 156], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 39], dtype='float32'),
            paddle.static.InputSpec(shape=[39, 156], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp11(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 30], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 30], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp12(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([30, 120], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[30, 120], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp13(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([512, 12544], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([12544, 1024], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[512, 12544], dtype='float32'),
            paddle.static.InputSpec(shape=[12544, 1024], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp14(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([512, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1024, 1024], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 1024], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp15(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([145, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[145, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp16(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([145, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[145, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp17(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 2, 2, 7, 7, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 2, 2, 7, 7, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp18(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([4, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[4, 2304, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp19(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([4, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[4, 576, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp20(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([54, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[54, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 1000], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp21(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([54, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[54, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 1000], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp22(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 1025, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 1025, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp23(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([22, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[22, 2048], dtype='float32'),
            paddle.static.InputSpec(shape=[2048, 1000], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp24(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 2, 2, 7, 7, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 2, 2, 7, 7, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp25(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1960, 16, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 48], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1960, 16, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp26(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1960, 16, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 24], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1960, 16, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp27(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([171, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[171, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 60], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp28(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([171, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[171, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 240], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp29(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 320, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 384], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp30(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 16384, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 16384, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp31(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([54, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[54, 198, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp32(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
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

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp33(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
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

class PrimitiveOp34(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp34(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 8, 8, 7, 7, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 8, 8, 7, 7, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp34()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp35(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp35(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 1, 1, 7, 7, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 1, 1, 7, 7, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp35()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp36(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp36(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 15], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp36()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp37(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp37(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[15, 60], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp37()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp38(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp38(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp38()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp39(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp39(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp39()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp40(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp40(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp40()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp41(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp41(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp41()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp42(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp42(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 672], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 168], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 672], dtype='float32'),
            paddle.static.InputSpec(shape=[672, 168], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp42()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp43(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp43(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 168], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([168, 672], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 168], dtype='float32'),
            paddle.static.InputSpec(shape=[168, 672], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp43()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp44(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp44(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 512, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 512, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp44()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp45(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp45(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 512, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 512, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 512], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp45()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp46(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp46(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 672], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 168], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 672], dtype='float32'),
            paddle.static.InputSpec(shape=[672, 168], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp46()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp47(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp47(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 168], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([168, 672], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 168], dtype='float32'),
            paddle.static.InputSpec(shape=[168, 672], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp47()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp48(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp48(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 4, 4, 7, 7, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 4, 4, 7, 7, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp48()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp49(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp49(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 2, 2, 7, 7, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 2, 2, 7, 7, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp49()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp50(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp50(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 2048, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([160, 160], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 2048, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[160, 160], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp50()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp51(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp51(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([22, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[22, 2048], dtype='float32'),
            paddle.static.InputSpec(shape=[2048, 1000], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp51()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp52(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp52(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 784, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp52()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp53(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp53(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 384], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp53()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp54(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp54(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 1, 1, 7, 7, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 1, 1, 7, 7, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp54()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp55(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp55(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 1174, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 1174, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp55()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp56(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp56(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp56()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp57(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp57(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp57()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp58(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp58(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp58()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp59(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp59(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 1024], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp59()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp60(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp60(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 784, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp60()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp61(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp61(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 384], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp61()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp62(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp62(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 40, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 6625], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 40, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 6625], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp62()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp63(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp63(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 1025, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 1025, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp63()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp64(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp64(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([22, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[22, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp64()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp65(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp65(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([22, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[22, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp65()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp66(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp66(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 4, 4, 7, 7, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 4, 4, 7, 7, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp66()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp67(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp67(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[36, 9], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp67()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp68(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp68(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[9, 36], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp68()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp69(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp69(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 8192, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 8192, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp69()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp70(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp70(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([145, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[145, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 60], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp70()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp71(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp71(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([145, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[145, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 240], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp71()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp72(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp72(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 512, 8192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8192, 512], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 512, 8192], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8192, 512], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp72()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp73(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp73(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 8192], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 8192], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp73()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp74(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp74(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[4, 9216, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp74()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp75(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp75(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([22, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[22, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 60], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp75()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp76(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp76(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([22, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[22, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 240], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp76()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp77(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp77(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 32768, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 32768, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp77()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp78(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp78(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([6, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[6, 144, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp78()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp79(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp79(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 1248], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1248, 312], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 1248], dtype='float32'),
            paddle.static.InputSpec(shape=[1248, 312], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp79()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp80(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp80(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 312], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([312, 1248], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 312], dtype='float32'),
            paddle.static.InputSpec(shape=[312, 1248], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp80()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp81(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp81(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([54, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[54, 197, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp81()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp82(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp82(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp82()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp83(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp83(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 1536], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp83()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp84(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp84(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 2, 2, 7, 7, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 2, 2, 7, 7, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp84()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp85(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp85(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([22, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[22, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 15], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp85()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp86(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp86(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([22, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[22, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[15, 60], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp86()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp87(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp87(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 2048], dtype='float32'),
            paddle.static.InputSpec(shape=[2048, 1000], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp87()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp88(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp88(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp88()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp89(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp89(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp89()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp90(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp90(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 624], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([624, 156], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 624], dtype='float32'),
            paddle.static.InputSpec(shape=[624, 156], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp90()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp91(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp91(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 156], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([156, 624], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 156], dtype='float32'),
            paddle.static.InputSpec(shape=[156, 624], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp91()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp92(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp92(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([171, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[171, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 15], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp92()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp93(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp93(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([171, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[171, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[15, 60], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp93()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp94(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp94(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 92], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([92, 23], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[92, 23], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp94()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp95(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp95(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 23], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([23, 92], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 23], dtype='float32'),
            paddle.static.InputSpec(shape=[23, 92], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp95()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp96(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp96(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([171, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[171, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp96()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp97(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp97(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([171, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[171, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp97()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp98(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp98(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([145, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[145, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[36, 9], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp98()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp99(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp99(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([145, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[145, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[9, 36], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp99()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp100(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp100(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 21504, 1, 91], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([91], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 21504, 1, 91], dtype='float32'),
            paddle.static.InputSpec(shape=[91], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp100()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp101(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp101(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([145, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[145, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp101()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp102(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp102(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([145, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[145, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp102()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp103(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp103(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 200, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 192], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 200, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 192], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp103()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp104(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp104(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 65536, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 65536, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp104()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp105(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp105(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 1024, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 1024, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp105()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp106(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp106(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 3136, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp106()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp107(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp107(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 192], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp107()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp108(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp108(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 2048, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([320, 320], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 2048, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320, 320], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp108()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp109(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp109(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([145, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[145, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 15], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp109()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp110(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp110(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([145, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[145, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[15, 60], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp110()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp111(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp111(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 320, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 384], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp111()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp112(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp112(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 4096, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 64, 4096], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 4096, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 64, 4096], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp112()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp113(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp113(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 512, 4096], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4096, 4096], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 512, 4096], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4096, 4096], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp113()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp114(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp114(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 1, 1, 7, 7, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 1, 1, 7, 7, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp114()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp115(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp115(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([4, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[4, 144, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp115()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp116(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp116(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 577, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 577, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp116()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp117(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp117(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 768], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 50, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 768], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp117()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp118(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp118(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 8192, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 64, 8192], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 8192, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 64, 8192], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp118()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp119(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp119(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 512, 8192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8192, 8192], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 512, 8192], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8192, 8192], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp119()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp120(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp120(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 8, 8, 7, 7, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 8, 8, 7, 7, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp120()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp121(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp121(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 32768, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 32768, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp121()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp122(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp122(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 4, 4, 7, 7, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 4, 4, 7, 7, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp122()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp123(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp123(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 784, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp123()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp124(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp124(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 384], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp124()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp125(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp125(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 960], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([960, 240], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 960], dtype='float32'),
            paddle.static.InputSpec(shape=[960, 240], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp125()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp126(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp126(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 960], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 960], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp126()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp127(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp127(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([22, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[22, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp127()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp128(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp128(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([22, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[22, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp128()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp129(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp129(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([171, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[171, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp129()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp130(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp130(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([171, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[171, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp130()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp131(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp131(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 8192, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 8192, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp131()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp132(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp132(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[6, 9216, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp132()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp133(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp133(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([86, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[86, 197, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp133()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp134(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp134(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([22, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[22, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[36, 9], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp134()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp135(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp135(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([22, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[22, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[9, 36], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp135()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp136(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp136(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp136()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp137(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp137(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp137()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp138(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp138(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([22, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[22, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp138()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp139(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp139(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([22, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[22, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp139()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp140(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp140(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 4096, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([320, 320], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 4096, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320, 320], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp140()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp141(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp141(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 640, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 192], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 640, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 192], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp141()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp142(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp142(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 768], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 160, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 768], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp142()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp143(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp143(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp143()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp144(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp144(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 1536], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp144()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp145(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp145(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 784, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp145()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp146(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp146(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 384], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp146()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp147(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp147(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([145, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[145, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp147()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp148(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp148(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([145, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[145, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp148()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp149(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp149(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 8, 8, 7, 7, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 8, 8, 7, 7, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp149()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp150(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp150(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([4312, 16, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 48], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[4312, 16, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp150()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp151(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp151(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([4312, 16, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 24], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[4312, 16, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp151()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp152(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp152(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([171, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[171, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[36, 9], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp152()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp153(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp153(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([171, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[171, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[9, 36], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp153()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp154(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp154(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([171, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[171, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp154()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp155(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp155(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([171, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[171, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp155()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp156(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp156(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp156()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp157(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp157(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 1536], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp157()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp158(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp158(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([86, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[86, 198, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp158()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp159(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp159(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 4, 4, 7, 7, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 4, 4, 7, 7, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp159()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp160(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp160(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 1024, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 150], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 1024, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 150], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp160()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp161(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp161(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([6, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[6, 2304, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp161()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp162(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp162(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp162()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp163(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp163(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp163()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp164(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp164(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 1024, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 1024, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp164()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp165(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp165(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 1024, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 1024, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 512], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp165()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp166(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp166(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 872], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([872, 218], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 872], dtype='float32'),
            paddle.static.InputSpec(shape=[872, 218], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp166()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp167(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp167(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 218], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([218, 872], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 218], dtype='float32'),
            paddle.static.InputSpec(shape=[218, 872], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp167()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp168(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp168(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 3136, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp168()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp169(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp169(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 192], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp169()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp170(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp170(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 3136, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp170()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp171(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp171(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 192], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp171()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp172(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp172(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 3136, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp172()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp173(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp173(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 192], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp173()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp174(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp174(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp174()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp175(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp175(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[11, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 1536], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp175()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp176(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp176(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([72, 18], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 72], dtype='float32'),
            paddle.static.InputSpec(shape=[72, 18], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp176()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp177(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp177(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([18, 72], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 18], dtype='float32'),
            paddle.static.InputSpec(shape=[18, 72], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp177()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp178(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp178(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 60], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp178()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp179(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp179(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 240], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp179()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp180(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp180(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([86, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[86, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 1000], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp180()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp181(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp181(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([86, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[86, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 1000], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp181()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp182(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp182(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 1024, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 1024, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp182()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp183(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp183(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp183()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp184(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp184(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp184()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp185(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp185(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 19, 32768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 32768, 512], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 19, 32768], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 32768, 512], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp185()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp186(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp186(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 1024, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 1024, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp186()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp187(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp187(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 1024, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 1024], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 1024, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp187()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp188(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp188(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([390, 3136], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3136, 1024], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[390, 3136], dtype='float32'),
            paddle.static.InputSpec(shape=[3136, 1024], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp188()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp189(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp189(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([390, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1024, 1024], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[390, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 1024], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp189()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp190(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp190(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 872], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([872, 218], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 872], dtype='float32'),
            paddle.static.InputSpec(shape=[872, 218], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp190()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp191(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp191(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 218], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([218, 872], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 218], dtype='float32'),
            paddle.static.InputSpec(shape=[218, 872], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp191()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp192(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp192(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 8, 8, 7, 7, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 8, 8, 7, 7, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp192()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp193(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp193(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([43, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[43, 2048], dtype='float32'),
            paddle.static.InputSpec(shape=[2048, 1000], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp193()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp194(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp194(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 1024, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 150], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 1024, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 150], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp194()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp195(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp195(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([6, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[6, 576, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp195()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp196(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp196(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 21, 16384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 16384, 512], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 21, 16384], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 16384, 512], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp196()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp197(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp197(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 197, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 197, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp197()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp198(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp198(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 197, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 197, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp198()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp199(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp199(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 16384, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 16384, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp199()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp200(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp200(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([10, 25, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 37], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[10, 25, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 37], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp200()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp201(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

class TestPrimitiveOp201(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 4096, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([160, 160], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 4096, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[160, 160], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp201()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

if __name__ == '__main__':
    unittest.main()