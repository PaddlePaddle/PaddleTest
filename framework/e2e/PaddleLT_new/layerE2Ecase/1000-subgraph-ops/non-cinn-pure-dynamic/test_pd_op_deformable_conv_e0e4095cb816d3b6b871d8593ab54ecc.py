import os
if os.getenv('FLAGS_cinn_new_group_scheduler') is None:
    os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
if os.getenv('FLAGS_group_schedule_tiling_first') is None:
    os.environ['FLAGS_group_schedule_tiling_first'] = '1'
if os.getenv('FLAGS_prim_all') is None:
    os.environ['FLAGS_prim_all'] = 'true'
if os.getenv('FLAGS_prim_enable_dynamic') is None:
    os.environ['FLAGS_prim_enable_dynamic'] = '1'
if os.getenv('FLAGS_enable_pir_api') is None:
    os.environ['FLAGS_enable_pir_api'] = '1'
if os.getenv('FLAGS_cinn_bucket_compile') is None:
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
    if enable_cinn is None:
        return False
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

def ApplyToStatic(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        input_spec=net.get_input_spec(),
        build_strategy=build_strategy,
        full_graph=True,
    )

class InstanceTrait:

    @classmethod
    def instance(cls):
        if cls.instance_ is None:
            cls.instance_ = cls()
        return cls.instance_

    @classmethod
    def static_instance_with_cinn(cls):
        if cls.static_instance_with_cinn_ is None:
            cls.static_instance_with_cinn_ = ApplyToStatic(
                cls.instance(),
                use_cinn=True
            )
        return cls.static_instance_with_cinn_

    @classmethod
    def static_instance_without_cinn(cls):
        if cls.static_instance_without_cinn_ is None:
            cls.static_instance_without_cinn_ = ApplyToStatic(
                cls.instance(),
                use_cinn=False
            )
        return cls.static_instance_without_cinn_


class CinnTestBase:

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

    def train(self, use_cinn):
        if GetEnvVarEnableJit():
            net = self.prepare_static_net(use_cinn)
        else:
            net = self.prepare_net()
        out = net(*self.inputs)
        return out
    
    def prepare_data(self):
        self.inputs = self.get_inputs()
        for input in self.inputs:
            input.stop_gradient = True

    def prepare_net(self):
        return self.get_test_class().instance()

    def prepare_static_net(self, use_cinn):
        if use_cinn:
            return self.get_test_class().static_instance_with_cinn()
        else:
            return self.get_test_class().static_instance_without_cinn()

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



class PrimitiveOp_a1606904a73af1d6281c9493645ba3a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7fc197e015d50c034dbd30b53589bdeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3530efa6dca48e27b61e17762f52c9f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_088bfa30ecd1f59af90dc557cde00b19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c6f642d087708a7fa7ba27fda58dce5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01a617110cce7d282558f805d038567a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1513c6a84d3e4d420762f929f534843f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6a10496ce5a2c81ed0ff75a5b5430a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7804dd830735527a2bc939ba61512b6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2bf66c984705437d1aefdb58f4f20ae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_584cff26f11436fb5cb41e5142d07258(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_51b8a1d6f18c3bf80bc7ea4f0f55f6f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d5c43c0eec99a44c9066934172a6b31a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7158ae6f1005d7d8bff1f7f9b21b3e06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f89ef3b50c3422cd6a0c40eaaf60bb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_de8c445ec00867d006ae5666fc394224(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa26798e7150cb19c3815a706ed27a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8bf88f1849da251e3c61daabac43048(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b25284c3584e3499a798aaab00a469c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a26dd99fd3913ca04756bebcc36501cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a99e9b2693edb8556a5a00886c5db4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_737d57418deeb8c054baf214d4846ecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d681995f2e7992ae161e71bd6cfd41e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 192, 288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6193cd8384fb65fb19fcd6865bece0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6b527750b73723baa1475981cde43d0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_709bb3a033fb5e0444343ba179c89700(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a79369cb480a3a9a50dab2dcf0ea4f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9becfca80c54820003464c260700dd83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [0, 0], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b9bb7cc8b4ffe9ac6cee1a0ecea3ab5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9becfca80c54820003464c260700dd83
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 120, 200], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ded267895792951420065a655bdbe93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72ccf3426abc71a0a8f3f55be7c7c612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04381ab63fed2cb7518d1cebd206fca4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd5d69ad3c1bb6e3c8c007ae1da49c70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a0f10c5b4fd77fe741b763b3132347e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c39f84d059b35bd1ec46e15d7b63b19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 14, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_569662d2eea7e2ba9ce301a0d2968ca2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9becfca80c54820003464c260700dd83
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()