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



class PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_26343443080d758f96161296cf55a9bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 24, 36], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0cd8fa81dbceffe2fa6eb67e0e8251b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 24, 36], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c1b0b571f9c2adee32b7729824698151(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c4d10c29014d08ebf56481280ac28cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b567806096c53cc546bc2cb3a0035a9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c1dc6c44379870267a3fb8ae79cfc3a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_84e135badc4506c9451f98bf9dee2a7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1dc6c44379870267a3fb8ae79cfc3a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_da232fd767b12c6862173c2137aa3164(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c15577731fdf8ec6bf78255366805cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c1067b803455a4d26e27c3b3d2d2cb61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1a0cb8bbdb8b4e3587ca5b2c55cf8c12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ad42532efac928a0712f56a77451e100(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8da493b34e8c1abf1f1e9d0eb6b5e2d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a599492113148c5ff9d0cbc7fcad219c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_07721f4be658b6fd4e0cba8f51768fa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0ed8231633974073bb409cd0686f7d06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_17146876e859c3b4cfa417169adee32f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a4ddd161d0f6e22bc8a6aa75f863ac3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1960, 4, 16, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d3cb78198c30907568de0cdc7d7eefd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1960, 4, 16, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_31612df5bc87584964e2658561c981af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7fab95d5898dc8fce7a66f810d20bcef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9befa6c0e41c0ed7fc35b1a758d87018(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aa1fb75cbb356aea67b7a2804b15d961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d5ff7c9b9da79790077f6d7a4a9f9bb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_359ef3803c3bb9ca6e270dd236cffb37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_594b1cd5f79b3e45c2433d9d9e412bca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a787bbb866b9f0e5c0193f98a055b050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_83648bc58748dad2ba5b03653b8c0273(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_723a4ef953739e0a3d3aff578f77e120(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9ac3cabacb3a5df5103927c97e0e1629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1a8495d1367805f7e63db936d740c84b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b0451d4c9d89f8e002456f1c6608edc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_99884c4cd7ad608f97a1badf2f78c943(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5a19c8c4de0a5cf06dbcaa551f7f7139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_507cab51751e3430273550eb3d93a585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d51d58ca2367fe2c9101346a932b1d27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 76], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_385177d1fad1d12fe2c45fdb9f54c5de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 76], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e664ad1b2a77758f97f3ab99d99cf1bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 76], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_355565a78683b76e915f7fdb1ebb0dac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 576, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6b491a031726f59e51da117b7532484a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 576, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_09a722e7f2ffd14f3b4c1eed8996f2a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 576, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c3ffcb0eee5ff072c522036e95c8b5d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 576, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d1f9ff74925ff1158515096450193484(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 288, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6f0c7aeda581da09bc8f1dead009762c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 288, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aadc92f2caada05cb6df27ba6b325b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 288, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_de90c70e54174d2b41e6ac0d22af98ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 288, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_edd86ea8cf9fd67f7364669ecda080d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_efd7a92b139c75edf2ebac383e43c255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c8507902b9f3d27661ab46da4af6b0ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_06c669b135a6974603e9be4001b64987(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3a1dddab2e841cc3aff6500aabb30c9c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f56224d4777e2f39bad94d238afde99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a1dddab2e841cc3aff6500aabb30c9c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.46424782276153564, 0.4705996513366699, 0.42634397745132446, 0.35580918192863464]], [[0.16565507650375366, 0.09719990193843842, 0.16619716584682465, 0.37486857175827026]], [[0.24447554349899292, 0.17953190207481384, 0.07921028137207031, 0.42684468626976013]], [[0.4760080575942993, 0.08616776019334793, 0.3810620605945587, 0.0039510601200163364]], [[0.25089192390441895, 0.4652281701564789, 0.40618160367012024, 0.3080861270427704]], [[0.32649412751197815, 0.18960219621658325, 0.2721419632434845, 0.15155068039894104]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e4407b505c98b08db49c929aef0df5e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a1dddab2e841cc3aff6500aabb30c9c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.46424782276153564, 0.4705996513366699, 0.42634397745132446, 0.35580918192863464]], [[0.16565507650375366, 0.09719990193843842, 0.16619716584682465, 0.37486857175827026]], [[0.24447554349899292, 0.17953190207481384, 0.07921028137207031, 0.42684468626976013]], [[0.4760080575942993, 0.08616776019334793, 0.3810620605945587, 0.0039510601200163364]], [[0.25089192390441895, 0.4652281701564789, 0.40618160367012024, 0.3080861270427704]], [[0.32649412751197815, 0.18960219621658325, 0.2721419632434845, 0.15155068039894104]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5fe355694916316f2f0a4d9edde18b3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a1dddab2e841cc3aff6500aabb30c9c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.46424782276153564, 0.4705996513366699, 0.42634397745132446, 0.35580918192863464]], [[0.16565507650375366, 0.09719990193843842, 0.16619716584682465, 0.37486857175827026]], [[0.24447554349899292, 0.17953190207481384, 0.07921028137207031, 0.42684468626976013]], [[0.4760080575942993, 0.08616776019334793, 0.3810620605945587, 0.0039510601200163364]], [[0.25089192390441895, 0.4652281701564789, 0.40618160367012024, 0.3080861270427704]], [[0.32649412751197815, 0.18960219621658325, 0.2721419632434845, 0.15155068039894104]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f68cd2f655d9f2041ed6aa0ccec0940d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a1dddab2e841cc3aff6500aabb30c9c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.46424782276153564, 0.4705996513366699, 0.42634397745132446, 0.35580918192863464]], [[0.16565507650375366, 0.09719990193843842, 0.16619716584682465, 0.37486857175827026]], [[0.24447554349899292, 0.17953190207481384, 0.07921028137207031, 0.42684468626976013]], [[0.4760080575942993, 0.08616776019334793, 0.3810620605945587, 0.0039510601200163364]], [[0.25089192390441895, 0.4652281701564789, 0.40618160367012024, 0.3080861270427704]], [[0.32649412751197815, 0.18960219621658325, 0.2721419632434845, 0.15155068039894104]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_73e2c1eda6d2b1f25847e2f2dcf82998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1dc6c44379870267a3fb8ae79cfc3a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_353e02bff50d0847315f33b4f4a926ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_88b22e06c9de72b4e662f6973d10c3aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_01421383b8fec4aa3df46125647e5964(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_95c8567f1f9ba78a5a8184fb70ff00c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7eea895fa4a5e19858d492444f55e1ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6706ffb0427efdb218252409bd0dd00a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0eb7654af72be91a45f8ac87f19b5570(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cef4a74ab6714b0051e9f55d9bc705f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bfd3f2e6bf7a0dfecccbf64a40b083ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2c042f3033d0400f79e7454b3c10df59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_df2b0c4abbcabc5bac554787de13a519(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e940c981b697ad0f8314d9dbe82efee8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e43576626e0273b984300199b66ce14b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([128, 320, 8, 4], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_63a4069116f059fc216dd38bf3bef5be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([128, 320, 8, 4], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d43893b70a99464177a88d0a1f5e84e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([128, 320, 8, 4], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f09c60ed1d085f8a6116174e4ced021a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d79fdd13a82b5b6d214a86bee87410f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ef7b05ed320a3f559b1241673f23c1f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 7, 7, 768], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_865bb47aee823c064646652f24593733(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_11f968dc90c1f67efa1e881ab229152d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9b0a2fad1dd46e23532ff28d5702117c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dcc7396e2e8b3f55c022051958acd8fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_062710a33fdcc9cd68500a4fbd606b51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_687a987ef1e2a76321df8a32db547e73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8af1a9f5e67af02e517bf230635cd70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f1196c6a5d640ecd42952ab333074e57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fe222c2c3e50d3c716e5153da6624a40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 128], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_93dc371b4a74c5f859c6b1774c19a573(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 128], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_23f4f9e49fdf739f39291169c70fa9d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_951e113f94036b4ecf3c89db4c384109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_680fae4a2beb4f3970f5aff5bf9842e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e5ae3f87fc06dda06665e94fb189f39b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_62333853b48770c3d6e42854936f8f3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0a6a7262875efefd17c06bff93027342(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_141453b423f07980f8b09b85ea549cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_013107fb168448e42e884a7cf79b0a43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7475a782e3d01b8d9fd0e3aa5ac4b3e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1999b914a42d815f236718a111f469f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 7581, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5ea2cebf464b45387b7676210436030f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 7581, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4b9d34d928a067d254e0600bc686b54b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 7581, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_27d70d113e3dcef97f95fbe4a4f6a6db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([528, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c0dbd04e9aea8bb355a69ab32a11e75b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([528, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ba365e1613427193a08d57fa7b8bf9ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([528, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_77b7f41391c620ad46e26e6b8cb6387c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([528, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bf738b4fbffca3c095bf611969f83b14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 25, 38], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_df98113b578af526122ce75ed8fec949(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 25, 38], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5ab59cfe90b683e2142d142c397fd26d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_93dbd4debcf264e16d6f46dd29c8d919(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_93b7397288b4f4ac776a6c22091a6993(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5434e051e3edcf84daaaa512b3d6d39c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 288, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c6d9ca0927e55170c7eb379b8bc6f4b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 288, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0cbaa979f6ee9032ac3ff6d507160fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 288, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cc8c0d6655971f0301eced77a46bf125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 320, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ae2f86d568222948672cb13b04f50102(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 320, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e342e553454c7122d1ec3a2d13d02c8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 320, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aa040afbbf14ff7687bf2d5a1be364d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4725, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4de6c759462d25a4ad562fee98617a9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4725, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_763865938955e68edf0f212d3bd77c59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4725, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c73071a94d83c41c4a95437d9fd59d99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5f12bfea29b1a5e93eef8bda2345058a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f589375138a57627e37d136777b9604d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_519e1e1fe08f16eddb905fdc69abe69a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 577, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7daeed48b1e488d1506cd7e11c779c24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 577, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bf7797c978a7ac330d4201010e4f57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a71df670149d60c96d89e53182833a70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eb79a3483f64cd5cfc7998c019ce4a77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_de1f707276ea0c63e11f75d0780a093f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_25ae664645a97d3afb9da697e397a16d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_54dfb995f1670595f22e02b85d6f43f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ffe45475833080c8bc9015af99e291b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([64, 64, 32, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d7ea9d62972dcbfbdeb700811e923309(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([64, 64, 32, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_760e33d02b881ebce2d688aeced22b0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([64, 64, 32, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0b50df8c4d711ea6681b9d56c2d27735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 10, 6, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_39720f18ab2c0ee2d0f961541745df8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 10, 6, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0ff492c7a79faadd7de2d8f8a66ff1ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([384, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4b33aa34b4315fb6478255d061f789da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([384, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9ee1b13f66a38af803c9b0f16d62c12d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([384, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_078d9c24ab6f9f8ff2942e2d68db663f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([384, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f063b26586463467b6017d2437cefecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_888a4d44ce1e118ec21826693ead9367(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_216bf0c201b22fd7e52b798a1221b7e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 28, 28, 192], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_53c2f9ffe51938b194bdc249a66ca998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_46831ba7b4a5fa97a7ab3f3bc80b2938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3c3335fff00e7d57182da9ef7cf4982d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c2162eee84a0ab6f3804ed6d86e4e116(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_03acbb48925a7203adedb3c099008007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_23f4f9e49fdf739f39291169c70fa9d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_951e113f94036b4ecf3c89db4c384109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84ebbde56a6ca2a45748f88ed8924aa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1fe1bdac5efcc807fbc8170e14995753(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7de15fa9768c395471dab8992141a1d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bf19962a7733d8a8c7b8ebd09f8f04e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1b963c3888a37cffb34d7b2bf00dca89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e8ed6bf80304136d444544eb3b9c37bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 8400, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b9eb4dbc8958b35bd98c0dd6f6bf0886(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 8400, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2fe4b491aa6b5d65117c4628aebd8b75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 8400, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4b10bb1eb340a5890a58b414ff97ebfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f582bf2258e548a9f80f82fbe201aa47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_424d6955898f30e5539b0e6dd6423e96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ff98725895c55a14690102d39f5e521c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dfda211df93d29538fff9d77e7340d67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 20, 30], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3fb3abcac4f791e00a5d0ffbd9c0ebae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 20, 30], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d028a5666cbd41142cd55b1caa1cc891(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a152cdb6f5fb6f9371002938676a730b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ef602eda349a34569b3739a1ac78ca8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_edd5e7e719d709838d1d61ca30ee05e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8683285b08b5f64608e70e86829b378f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9560ea257e3b7e484278f188777ea2e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ccea88b68a85ee1d3b93ae071dd90c61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_59008423372c28d07362d6e7592a12ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1e9c96ddbbe9c050e4544e833fd85640(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_57a07320e5adbad248b8b50272a02a48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e440a07e0627e584d65bd07c10444d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f31681bcb7bd5ed0d87e74ae80f8c16c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b9444c83d9326753bcc5eeae1e05364a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_17f460fb3f0eb1bcef0676bba3b0234d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0fda9a00d40957b04b84044912c4711e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c852fc07f0af2b879974671150e5ad32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b2f019a47742cd0fff611819f2f54c05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8b21ee1ed65ba17036a312f6ec9d22a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_062710a33fdcc9cd68500a4fbd606b51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_687a987ef1e2a76321df8a32db547e73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8af1a9f5e67af02e517bf230635cd70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f1196c6a5d640ecd42952ab333074e57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d391a769bd790b26195ea7444177ac02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4310a927c09bea769a9269cdc76cedfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a82b3142da0b4d360eaef78ccd2d1d9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5905642d885065b305c32411cd278349(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c95850ace86ccc62a4cf1925a0ff8bf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c2eba266db7c82567621f76b113b595d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f9f5b6b70164fe050f44f071e301d54c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6e122009d1f1ae1e063d46c37a9af605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6432aa4a4596983e17e2648b50653550(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0f1549adeeda8280588e3484cacba120(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e3f0f980ce5b0e4a8f75235e512c59ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cbbe43281013a92f78816039a9812fc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1f0a993ff9bc2ddca05ab898f3fdb410(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7e1ccecc9aa5dacdbe14afa57be0f7ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_daca08fa2679e9300d7a0c06a62d541a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7410126d567ae6684926d8d8915286b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d2e88242d6a00ecc9455754923a5e7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9c31af418577084129a1ce33ad645c33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1c24ba3feb740289d113c1b3fa020448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_75221dbcb46398629b9df078914dc3b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c34f1c7b04f0daf088d588507e24fee9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_46bd6b0faf5552136245e0e446409a3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_91f7b6bec5e67d738959e30172103f57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7b392594ed9c25c6f60fc43da690efd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6314f517a774c5a246e514efc6941ea5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([20, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9179f8b252480a063449bc73467626b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([20, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_54087614fc161379771027df53a0e2ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([20, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6260de83d8fbbab3f2001df7d4380b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([20, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8d7842ab21705b736416975a4926d724(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a966fc7d735c473df83f9299053ce62f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6353595a19792ab678c8540723edf159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_817e3dd57de2bb83f29861be46d07d7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 97, 97], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9f78f1df3721a9c75316da6a9a7d6f55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 97, 97], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_52c1ecf858aec2a603fcfe39b166594b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [-1, -1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a3d16c8e094cd385bcf2edd01a56d990(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52c1ecf858aec2a603fcfe39b166594b
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
            paddle.to_tensor([98, 99], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_98c79c7e24332be86218f4c7b5cb995f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 4312, 4, 16, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e6ced568b9ec20a933082aff657fedd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 4312, 4, 16, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_266ac6d4ae68446fbe97d27eebef11bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_231afa21e0f1a1a8ff0a066675fbf553(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_99884c4cd7ad608f97a1badf2f78c943(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5a19c8c4de0a5cf06dbcaa551f7f7139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_507cab51751e3430273550eb3d93a585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_496a896fc9d940b51187a487b3c4ad48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_813fed9687e418cb8553f9c4f504fca7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_baef98d121297b072d17cc8b6520dcf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 7, 7, 768], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b4c2c5edf304863e0edcfd5e12528f07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 1024, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a75153d42f54559ccd8ce3739240428b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 1024, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_716b11e7a0cc3cf59f867e451b113035(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 1024, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c279c9c3ede7f0c927c41cd4b5a82017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 256, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a5fb3bc2da58a6cf26c485f38ea95d27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 256, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_023eef1ed3d812b4e1fc1a43a309a678(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 256, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d35e96704abdb41ea9269a702c096830(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([576, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f69ad7a0f2e45f1cbc2db8a5f26bcfc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([576, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_57fd049da714db6930591f02f0ab4c57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([576, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_26cfb5915155836721f35465e88c10e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([576, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7bb02db1bb178e5251612efa407d62c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([22, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fc68ac0786ab027f05c7cb9ba3de41b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([22, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_79021b07a3d82e802a25d29b246900fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([22, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bb7bb632653b00bbc94fe6291865ee34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([22, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_919b072a24052a1516275cd9be1923a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([96, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cc0b64917c83f36f7201f542ca87f86e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([96, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f53707ccba20d5c777dcb759010f4ab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([96, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5a193d8f4ac30fcde7bf57f63beb5c1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([96, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_de199ce1636ab79d62bcb034c1145dc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c28c465c4ec13a9215851850054557e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_99a7b8d91714345fa2a6d628c0a6ecea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ecdb48d14b1b3d365badf8eacd767d02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7fb9ba6309595aff4258941579253231(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 256], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2eac4df153d887ea25fc4be5ba72d881(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 256], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7d925a27b4ae010b98819e765986d360(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a507cff3cf1dba7c739468162cfd862b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6432aa4a4596983e17e2648b50653550(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0f1549adeeda8280588e3484cacba120(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c1f6b2eeabfa03797627bfee41a4032d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f494a2733a352aa8b0b82c6b5a1af7d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6081c39fe008fad5234d438a111a26f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_42a9068d5402023d29b6b4035bf8e5b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d9165009134843ba97ca4e987d900e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3c85f072536f36f9f35dd9a99bb7212e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a6fd635843b6b96cf0f47c60c7c823e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([960, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_43fcaf12e5333b6fc65a63d35b42a4a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([960, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3327cb5a07fa0d968d78b715f1e0178e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([960, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c31d331aff068682f000afb179e13562(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([960, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ad42532efac928a0712f56a77451e100(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8da493b34e8c1abf1f1e9d0eb6b5e2d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a599492113148c5ff9d0cbc7fcad219c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_363128aa02166c5873f280d80661a9f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5881022ed9281acf261d44a195b9d5fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_008d954cb529c4236a31e9f752e00c37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 256, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e24ca639c53ae4414902b9468d2854a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 256, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a9235d0d068a0a8d418d939118c5ebaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([2112, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9d25c4a903c211342cba321856dd81a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([2112, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8184d93ae63ab2a199d3ba89d8c4f775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([2112, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fe08f03f899568a2ec09a1dd6e13f0ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([2112, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_85e5cc60d7099c03d56f68663f063fb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 72, 28, 50], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_00e477a68e4a35b8e7fad025609c2186(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 72, 28, 50], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7833fc6b44240a902819a5e48757e669(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 72, 28, 50], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ede928c40f986ae69c0a9456a1db8406(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 72, 28, 50], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4d6ce11cfd3eddd429b9d126146a60c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a1dddab2e841cc3aff6500aabb30c9c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.13873741030693054, 0.3854258060455322, 0.18275712430477142, 0.43969258666038513]], [[0.13893212378025055, 0.18957515060901642, 0.38497790694236755, 0.3787880539894104]], [[0.2273373305797577, 0.17694276571273804, 0.13323169946670532, 0.4903319180011749]], [[0.17077070474624634, 0.0017483800183981657, 0.19547641277313232, 0.11881936341524124]], [[0.2892541289329529, 0.26534363627433777, 0.37189048528671265, 0.18562017381191254]], [[0.25426748394966125, 0.07485723495483398, 0.42880791425704956, 0.12537863850593567]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_337ef5285ed60461dad8940064a21f2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a1dddab2e841cc3aff6500aabb30c9c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.13873741030693054, 0.3854258060455322, 0.18275712430477142, 0.43969258666038513]], [[0.13893212378025055, 0.18957515060901642, 0.38497790694236755, 0.3787880539894104]], [[0.2273373305797577, 0.17694276571273804, 0.13323169946670532, 0.4903319180011749]], [[0.17077070474624634, 0.0017483800183981657, 0.19547641277313232, 0.11881936341524124]], [[0.2892541289329529, 0.26534363627433777, 0.37189048528671265, 0.18562017381191254]], [[0.25426748394966125, 0.07485723495483398, 0.42880791425704956, 0.12537863850593567]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2530773cc90fac4e746718eb97d32d37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a1dddab2e841cc3aff6500aabb30c9c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.13873741030693054, 0.3854258060455322, 0.18275712430477142, 0.43969258666038513]], [[0.13893212378025055, 0.18957515060901642, 0.38497790694236755, 0.3787880539894104]], [[0.2273373305797577, 0.17694276571273804, 0.13323169946670532, 0.4903319180011749]], [[0.17077070474624634, 0.0017483800183981657, 0.19547641277313232, 0.11881936341524124]], [[0.2892541289329529, 0.26534363627433777, 0.37189048528671265, 0.18562017381191254]], [[0.25426748394966125, 0.07485723495483398, 0.42880791425704956, 0.12537863850593567]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9a59b4936748fd5a3e6413ac4c2529d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a1dddab2e841cc3aff6500aabb30c9c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.13873741030693054, 0.3854258060455322, 0.18275712430477142, 0.43969258666038513]], [[0.13893212378025055, 0.18957515060901642, 0.38497790694236755, 0.3787880539894104]], [[0.2273373305797577, 0.17694276571273804, 0.13323169946670532, 0.4903319180011749]], [[0.17077070474624634, 0.0017483800183981657, 0.19547641277313232, 0.11881936341524124]], [[0.2892541289329529, 0.26534363627433777, 0.37189048528671265, 0.18562017381191254]], [[0.25426748394966125, 0.07485723495483398, 0.42880791425704956, 0.12537863850593567]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_69290bf8229703e1327398f06ae99dab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4116, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_91faf3131fc437465dceae700df1b1b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4116, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b80bcd875e9e9098863dddd6ff2d0b3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4116, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15d3433429adf42d48390a15903c16dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cd6277ee13835399ac92102d0eb5c6e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2a557159f4849698befee003ec0da829(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f6ea1ba27e496a68e79e2f51584627e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_79f7fbb77c75b491aac4c49f099a249f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aeb41868dd6134f6b63a1333bd81bb0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5ad62ea5eea9f572832655a275989ab7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a645040f41b0c2d2b9abe31ebd80c2f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_399c50562d8e74a3cc6edd6bd8aada80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9387198cd162dd81d6054f5454e17756(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7483bdc2246b7d2bf2e229d342ed6aea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a692818d8862b2a49228617790067a14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_488a3801137311915557fe2a8b302896(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_69a12badb8a5d30f1e78e5787141156d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_763b63698bc254e95b55743e97e11043(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e4689e14cf630e511f4770acdea6d240(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d7d567a88332cba0e8825312936300ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_79501940c3a58ebf99ebfca727d8511d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bcf490da0763cac3fabb24df441122c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3dbfca7f3e08d7706d40884fdb778941(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1b0c2c346401246e81c7a62cc6cd51db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b449b770cba9d9631be77bcf57806b88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8573d8193ac5190f10c5f05a0ed50152(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6f0a7df86647f85eb45934584c057771(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8149e6ae47f2b9d370db0e0d6605c56b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7c94afeb274df7658f768e7fd386c6d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_acbe453259b1dfe2b9c98b8ab1a9878c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b1b753b934ae63a568db9025a7f4604b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5bf998f02d6b5f7d407f1a2d7e33f48d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5a7dfd31fceba688fd4f75243a146daa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d92402fe7255d339e01bce34c4965b18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6856ea2d8cdc98762dd6df5d52a28637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_59b8aa78459cbf8aa29f0c96a96c5c9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5a1778f13bc915d8f1544a0c648d03d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1032a22229459122ffebff6577144bf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1be3dd9a23ee7e0b39f57fc45e2ce05a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_617668f0aec0ff976dfbd8933167e8c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_406f424c08704eb2713eb9b4ec98fa68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3786fa8a9ef733034ee4a988628a87f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_828a3923415e09ee82f98632d1b1f17b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cb72e334928387817f4a2557bb92124e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_456340d75a6b0dce49c2bda65ff6786a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_777d67f3d82ca89241a75c0be3086fba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_730fd2c60edf736bd8b40f569ced0be7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9995d1885f7d1924d81c233c52195841(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2926ebc84ecdffe63812165bbdb7e7f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5ee9f5f5c94f991de765e1f1363de15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3c43a554909d5a69414d69be74448460(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_838e19ee74d3ba7deaa02b7ebd3f323f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_56be7672bc958b895e2be9c018ae7cf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4dd6ee3d229cd0f8789708a7d8eaf52b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fffe098cbab5da01c08d91b406e1b573(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15d3433429adf42d48390a15903c16dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cd6277ee13835399ac92102d0eb5c6e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_07113defb0f168370f9effdbb4ac1557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_45a8f71dcb79f3ea465a1ef358a7a55d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_062710a33fdcc9cd68500a4fbd606b51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_687a987ef1e2a76321df8a32db547e73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8af1a9f5e67af02e517bf230635cd70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f1196c6a5d640ecd42952ab333074e57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8683285b08b5f64608e70e86829b378f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9560ea257e3b7e484278f188777ea2e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ccea88b68a85ee1d3b93ae071dd90c61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6233ace6c524fb52f049aaaa1012e4e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 960, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_744be32e4ba10e8081989f79f3c346bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 960, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_56ef949e660926da665fc07ec1f93157(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 960, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3da6bc90e49ca6e5da032d288a8154b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 960, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d00cec04f5679181df0df0ecb00cf311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 480, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84bd5db1430796271a26365cd4424bb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 480, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_704e9eb7c52f1cc9c1afbaabf63357d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 480, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ad85c2a5e5aa3ab76d0334ab026244c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 480, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1057121f3bf04e686b4196a3fc81c170(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 240, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3529d7d3d0631402a0852b80880c542b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 240, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3ed430d650e20131d9789e22f4b2fe22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 240, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cd147984c29ae49f8941e940daa8c5b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 240, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d05f5d6534212a57558771633f8eb920(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 6069, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_586a8e3f9ceb0e8732c4f75321fe6ce7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 6069, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f19d1fc0eb16ebe8f6e9e7dee7bca3cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 6069, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e3f8c028fb7ebae21a02308abb0987ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_69aafb6f2c3e44d4fc099ed8322ee424(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ca6409ee02a9041c53047f5dc85f69a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 28, 28, 192], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6a1de0d1c36211589c7121d124a1e125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_53c2f9ffe51938b194bdc249a66ca998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_46831ba7b4a5fa97a7ab3f3bc80b2938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3c3335fff00e7d57182da9ef7cf4982d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_37405d3a51a266fe5350748ad4486ea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 8], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4024e78a74a044b4d2a85e7c597c3c11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_09e59c3ad40742665494ac31871dfc7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7035206c60b70d388825821db0ad2586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8827b52c65f5aab93a3cf4bd38d47bc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7f981a7d7b55767d3305aa8c2c191b5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ebde116a205f47f8a66622e34485a7c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_52e34b91ff282bec1f688703ddc0fd8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7a85f333df054c8622c88a445ca3af6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9400328adfe6f16d36846ab68bd4d6bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_749b9c5566c100a9206decc33391d2a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_baceb2f559f14ea30c275772cd54c587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_41cd072993634d83466cf06c7e81002a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cfd80272c709a47d593c8d5329a306f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aa012d6cad2c76f7926a2efc2f1ee6ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_25236e2ca720de46219ba13d049339d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e64e749420f10ce6d8b3abe06c3a9efa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f1df108ff113b28a644ffabe7b26ae83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6a1de0d1c36211589c7121d124a1e125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_de1f707276ea0c63e11f75d0780a093f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_25ae664645a97d3afb9da697e397a16d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_54dfb995f1670595f22e02b85d6f43f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_77dd19ec790112c037e45cb4c1f15513(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf889aca04dc0a63021315e91ed58f1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9275f025addfd7af5e2476e7a8200c6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a2baf0a464033315c792480303d8605b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_be735f5e453ef41042dc7c621c5615e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9970ad5cdc16992210108a3f20ba550c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be735f5e453ef41042dc7c621c5615e7
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 7], dtype='int64').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_97094cb729492fcd0844f1c2c54992f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be735f5e453ef41042dc7c621c5615e7
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 7], dtype='int64').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_16662624abf3085f7e6efca3e91c1e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 256], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5852b77d17bdb5905ce2864981e4b9c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 256], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_df17249d1a96daad1ba92f717df068db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d4373f35aef029af57b1ed8d2b58fdb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c6bbf0b8b5c13b9822db2b50a03fa7c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f6c3903b9467cd19b7b2c91b70ce6e09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15d609242be52a9de6737a6d27d29b0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c4d10c29014d08ebf56481280ac28cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b567806096c53cc546bc2cb3a0035a9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8bf919a7d4a94455278867058c3db924(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 464, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ab0bb084bda23cc6989b2ffdcc99ec82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 464, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7843932c486fa0720cc78358c877bc03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 464, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_92727b1c5d9f0fb86188c5b3136cf70e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 464, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8d7842ab21705b736416975a4926d724(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a966fc7d735c473df83f9299053ce62f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6353595a19792ab678c8540723edf159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_02588515735128b5ea1f44db7a0a4e0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_20316c1885ba6218527b86114677fdc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9596715a847b11610a731203fb450d75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0fda9a00d40957b04b84044912c4711e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c852fc07f0af2b879974671150e5ad32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b2f019a47742cd0fff611819f2f54c05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8b21ee1ed65ba17036a312f6ec9d22a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_062710a33fdcc9cd68500a4fbd606b51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8af1a9f5e67af02e517bf230635cd70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f1196c6a5d640ecd42952ab333074e57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0fda9a00d40957b04b84044912c4711e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b2f019a47742cd0fff611819f2f54c05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8b21ee1ed65ba17036a312f6ec9d22a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c085a3e5822ef141af68d6d052505378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7249aef4c20ab5fb0cf6ed7f3e491e85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84e135badc4506c9451f98bf9dee2a7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1dc6c44379870267a3fb8ae79cfc3a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5ab59cfe90b683e2142d142c397fd26d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_93dbd4debcf264e16d6f46dd29c8d919(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_93b7397288b4f4ac776a6c22091a6993(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_485d253e7026ccefdfe166c6c3d92aac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ffadb3bc9f885b927d04529dcf9378fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6d7e5344d36f07928e95e6fcc2966fa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_00993eca08e040fde6fd176600da0760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_93632e0928214b62aabcc3de219ed808(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_98f63e18dfeefe01a3730694623690ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1bc5c818ea80c53252b0d343885874b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c2dc354c40b58549330f9a4cd5a18e40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_69ee650042994306fc0502db93e74e6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ff3894152cbc42e00027bdf39a214e62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_67fb325a5bf980c543ba425c0b06726b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 14, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9ca5fef1c0a009c1763941384ab48a99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 14, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8c5a55690bf8d2ffafa9dc15e3e29648(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 14, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8e3c1042de66c383bcfd907449caf3ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 14, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1c24ba3feb740289d113c1b3fa020448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_75221dbcb46398629b9df078914dc3b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_690149a196559db16c3c70246b872950(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 128, 1, 1], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fe203ead6db672361748705a607fce36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 128, 1, 1], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_13595eea850c403714e92ce1f39a915a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6d6a57d7d21b088ce43cb6729bb18e7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f9f8ed10c5b7e7a3a572b8b99521dfcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_141453b423f07980f8b09b85ea549cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_013107fb168448e42e884a7cf79b0a43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7475a782e3d01b8d9fd0e3aa5ac4b3e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6647126b5e72f5bd909ab886a460307b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3d906a99580b4a299a86c53fc422b637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cfd80272c709a47d593c8d5329a306f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aa012d6cad2c76f7926a2efc2f1ee6ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_25236e2ca720de46219ba13d049339d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e64e749420f10ce6d8b3abe06c3a9efa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_de35199cae81aeba33ad909b879df588(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_915248c1bfd62e703195394c899a049f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_896c237ca05dc5374efebeea7160058d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fea6a12d2759bb8cafa840fa59f0a950(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 9261, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6b1c2e9325d52e1e5df90acc91b38ecb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 9261, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf8b05fb945dc9a991cc25a17f78a142(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 9261, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_13ad9aa2b0db49984c6c3b33fd061ece(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7db06de4c2c7dbf70852ba33a73a48ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_26a86d56a24e8e72ee7d1c594987b9e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_569745004c758fd840dc178518a88413(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_89675dbc21d6dc255f1baeeb748574db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_330f3925909892a5d2c3ee74a44bbf4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0eeeeedc4c138f710b236c953614a4b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0b3b5bedaae798921a915456d2f826c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2de58aac56d120db5e597f99c7a503d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f1371e7e5317edd5914c76d4fb407ba4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_551adf26886de435ed9f6283ee51f695(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f693c03b3d720e89520bcda862130606(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e1ad2a18cf03cb8ccc396a84ae9b054d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9aa33ed7ba56a6c5e5b3058b98b5a9a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a736e7929616c0569bee3bedb1c99f43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4c9e4efaed37e95c1fb2b22970b275a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bbdd41eb7f969a7c02560833cc3b9b5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 22, 6, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ba77c706c02469fcaafd81622170ce79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 22, 6, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ab73346dfc3a6b82730ab8ea35a38bab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([240, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4181ab0124414d255e29ebfd6a012295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([240, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1dc1520d01e22dd57cf775353718fd1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([240, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d000b6307082d89f8430323746be69a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([240, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ae0e4582ed85adc125f2dea03c724b41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fed61dac72791ba69c80a09d9e629146(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_abcf2faa8d9d59a5519b536280e2544c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5b9589715aa20557fe21129d000f03f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d2e88242d6a00ecc9455754923a5e7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9c31af418577084129a1ce33ad645c33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d2e88242d6a00ecc9455754923a5e7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9c31af418577084129a1ce33ad645c33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d2e88242d6a00ecc9455754923a5e7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9c31af418577084129a1ce33ad645c33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_942a2197e2635c0a7c43f4e433f412f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c3408c3810ee406853d692aec0e02a3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e3f8c028fb7ebae21a02308abb0987ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_69aafb6f2c3e44d4fc099ed8322ee424(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ca6409ee02a9041c53047f5dc85f69a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 28, 28, 192], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_46181c419f803df0f90583ba7a1c51e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a26e9068f1243555209d96d310283d8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d6eb5eb7e1291666b9e8313c4ab6700c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4723dd1c71e51069ff5d404f1ca2a10f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4cbeddf1c37ab09723024674ca36bc86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3801e53147af703ba44aa8e295ddfff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ce1c38bc8788900260ff7133cea9de65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3b24f1905405a70e40af5e5822224bd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_979bb1b726d93a27808f76c9a1cb0354(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_578efc777bb292a9384d3bed64d73fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7fc1cdac3cfba1bb3e518279bd64b24e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0662516b44473863f2f9e34a40d4b4dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5995541f983a28c2d9c85219c9e7dbb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2e0b89412df82f8f23b0b03694f14da3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eadf8dc5c574c1bb62e5c41d6f265f62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b270e3d08310f6a20559096a4a14c4bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5c77be29c34af66e2bb18641446b3ab9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e678125594bda81233eedd95a54cb717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 16, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_642c41ebe000e9b7373a93169e1c9144(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 16, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_798e088a710f2be462c1f11ef2bf3d9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 16, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a383908213191d1da03a16f9ff6383f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 16, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5b6b6527d37f44b8aa71d73652f659ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 8, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_90775abce73163e0881e518b4736adfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 8, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_094ce0ea439335bdc116177bb8f23e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 8, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6917049646f672019763db550c52ed38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 8, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6bc7c6aacc5b69b4aeddc3bad981184b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_94d1ec65a124ea6dd1f83ad1076461fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15691551afa1d7a4b2e45a467e3ba898(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2100, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e765d3d780df2b1b10e8bd1343edcd61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2100, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_010ae3bd5229a8250f20f67f83d0ecc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2100, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d2e88242d6a00ecc9455754923a5e7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_388ea0c3f3fe28966b64e2aea7ec3326(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_41545d7f6a02a5b77d644eb79e83726d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ebde116a205f47f8a66622e34485a7c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f222e4d3bba2017da376e8753189c87d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ebde116a205f47f8a66622e34485a7c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f222e4d3bba2017da376e8753189c87d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ebde116a205f47f8a66622e34485a7c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f222e4d3bba2017da376e8753189c87d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9fbe64a02c84fab3b9c0af6da9caa3e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9d17bfa1a26a37d3af2b4a411d531c28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_00993eca08e040fde6fd176600da0760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_93632e0928214b62aabcc3de219ed808(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_98f63e18dfeefe01a3730694623690ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0c35afb386ed4752ac22e5ffe8b48958(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 15, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1e4be859519395e7eaa121bc9629de63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 15, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_de1f707276ea0c63e11f75d0780a093f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_25ae664645a97d3afb9da697e397a16d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_54dfb995f1670595f22e02b85d6f43f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86ee7a039f62af77144d2223f1f829da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ae2fe8ebd42024e82855ba64b09893ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_37810b3a2a9a0876e76751fa91eee4b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_619cfc4a3d9275b2082d7c5706dd2eb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9837b1ae46df01b3576b5d5db4657105(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ddfd377aaa17c6cb05a2c72508785991(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86ee7a039f62af77144d2223f1f829da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ae2fe8ebd42024e82855ba64b09893ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_37810b3a2a9a0876e76751fa91eee4b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f9f5b6b70164fe050f44f071e301d54c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6e122009d1f1ae1e063d46c37a9af605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4f603a7df1201f1be27dbc9f194734ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c35c2fb49162c6178e6314af91a9ccfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6bde777f9a738bddb4a6e7f37d43297b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f0eff28a30784fd5f4f63680b30c8200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([44, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b8e98201ff6fe4ac09bb511686da2285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([44, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6525e544376c5d842fa784db4d1b6fdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([44, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9ca107adaa545722c582cf01cbf1f90f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([44, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_619cfc4a3d9275b2082d7c5706dd2eb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9837b1ae46df01b3576b5d5db4657105(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ddfd377aaa17c6cb05a2c72508785991(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e135e916358ff98da88da5942abc5b54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_de92d1294e1b799e270afae5b424ed0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9886bcd6bb322af693b53b32193f1740(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_58f9fbffe156a41f0f2780b54da2d792(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_13595eea850c403714e92ce1f39a915a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6d6a57d7d21b088ce43cb6729bb18e7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f9f8ed10c5b7e7a3a572b8b99521dfcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_73e2c1eda6d2b1f25847e2f2dcf82998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1dc6c44379870267a3fb8ae79cfc3a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d2e88242d6a00ecc9455754923a5e7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9c31af418577084129a1ce33ad645c33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_388ea0c3f3fe28966b64e2aea7ec3326(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_41545d7f6a02a5b77d644eb79e83726d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b24ed5cba7763faac3b0182275f4f368(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 512], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5a8295eb649de9ff2511f26388f4ec90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 512], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_34810a5cba2bd6ec4c545d09b6183e03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c6d62ca815072eda2a19ae5dd66c3557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c98d61a920ff9fe6c4c5ea7aae0782d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 11109, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_568a9c4ffd7c8a520904490d4acfb395(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 11109, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_adedab5f07dcfe4292ab88c03637dac1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 11109, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c15577731fdf8ec6bf78255366805cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c1067b803455a4d26e27c3b3d2d2cb61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1a0cb8bbdb8b4e3587ca5b2c55cf8c12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f09c60ed1d085f8a6116174e4ced021a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d79fdd13a82b5b6d214a86bee87410f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ef7b05ed320a3f559b1241673f23c1f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 7, 7, 768], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_46181c419f803df0f90583ba7a1c51e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a26e9068f1243555209d96d310283d8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d6eb5eb7e1291666b9e8313c4ab6700c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_496a896fc9d940b51187a487b3c4ad48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_813fed9687e418cb8553f9c4f504fca7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_baef98d121297b072d17cc8b6520dcf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 7, 7, 768], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bf19962a7733d8a8c7b8ebd09f8f04e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1b963c3888a37cffb34d7b2bf00dca89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_833f9250a3f496fe8d03a432680cf943(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c6f1efe88f1e10c4d1b2662f5a7a1977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_578efc777bb292a9384d3bed64d73fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7fc1cdac3cfba1bb3e518279bd64b24e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_967ced05864ec0d0bf7a8fbaa8282b4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([144, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8c992f9e83beba3c0adca8e827ece5f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([144, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b814eb47dd9f2b1274b96687476eecae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([144, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e0962324a18179f2b24f5601d4534c74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([144, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ff5e77916d874380cc8d2f69bd771ff1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e2da3a434e94da40afa61c6f628b0bdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4d0049cb6e837c0bb03382cef6dbda21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_239f542380b4161fe9717392d7ecec41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_353e02bff50d0847315f33b4f4a926ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_88b22e06c9de72b4e662f6973d10c3aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_01421383b8fec4aa3df46125647e5964(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_95c8567f1f9ba78a5a8184fb70ff00c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7eea895fa4a5e19858d492444f55e1ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6706ffb0427efdb218252409bd0dd00a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0eb7654af72be91a45f8ac87f19b5570(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cef4a74ab6714b0051e9f55d9bc705f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_059fc00acc086d50dc17586dfbef20d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 160, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9d7ac6aeaf86901103089074731adcf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 160, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8e344a637c28b1f39c20433beeb8c1ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 160, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b1d80fcad06c5f5e2565d197a5a7e80b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 160, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_39013c3c427d2ce54a6d2d7ffdcc7f5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ccacfa81020c698e36dbb93377f15164(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bb4114aa23180a3e9507c8e3d6a94bb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_46bfe095cdd00ee96cdbae6cd75b6644(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_266ac6d4ae68446fbe97d27eebef11bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_231afa21e0f1a1a8ff0a066675fbf553(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5d4b33ec66b0d813318a83a52f082edf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_07047a75e92b53e3649a512a49119cdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_780a3483550de9f33ef60c977dc371c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4fe3fdbb1a24fdc2b766b9f3d7066dfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_05153acc918568846dd6ed42bf00f0f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_56bc2aee05646037a1ddc26039564a8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_df2b0c4abbcabc5bac554787de13a519(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e940c981b697ad0f8314d9dbe82efee8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_37b43ecd81d39eb8ed8c29509b641686(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 512], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b737c474b74e75b5d937c925afeb5e10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 512], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2316edef301c8a3a92c668846a1b2331(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0e582633569159f250420d916d1f6da3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f063b26586463467b6017d2437cefecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_888a4d44ce1e118ec21826693ead9367(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_216bf0c201b22fd7e52b798a1221b7e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 28, 28, 192], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0f3a03d4c5ba900073e5ae16b12befdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2e4bc642d40115722334055019c2fcb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d5ff7c9b9da79790077f6d7a4a9f9bb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_359ef3803c3bb9ca6e270dd236cffb37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b3d9edb3f24b51103edbea26e33c9861(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 232, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c5e4972735d3fd805d1aec42ffec132f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 232, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1b98a29c7e1d00c4cc75dc23cd8ba7b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 232, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e284891509fcc85de61fe98f4d8f4599(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 232, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d9165009134843ba97ca4e987d900e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3c85f072536f36f9f35dd9a99bb7212e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8683285b08b5f64608e70e86829b378f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9560ea257e3b7e484278f188777ea2e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ccea88b68a85ee1d3b93ae071dd90c61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1f66a943ba858f53df29561204eb00c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2f1e3e3b3de898636cf341e4e326990e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_df9181e30ea624cd346347888ddb3a90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_27328cc7e974760da2bd12a87f545eb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6181a52d196a1c21fed53dab4f619d18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eb99dc030002f0d659ee8a80bee872bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3b40d04b27db6304d54bd0f75b69aee8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fd36fbe242527e76d7acdf8f221dd519(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_31f65f94f9516e1eea294c9e230b46d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 97, 97], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_761bdbc5a2adc5c0d2188ee26e7c20b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 97, 97], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_809b4a588b273919882f17386f49e874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ca0fb268ab98b429b8c115964066c2d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ece2b0aef93b8e2317508b42753d10a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_44f1447d66e1e0555b313698d90cba60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4bece61fcc4d1190287719372c40956c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e3781d3d98c43fdfecfd28982a6d792c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_97a631a67c27c2b1156bdeb1f58ca64f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1220602a1bd5502c4ba6e7f260019483(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ebde116a205f47f8a66622e34485a7c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f222e4d3bba2017da376e8753189c87d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_52e34b91ff282bec1f688703ddc0fd8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7a85f333df054c8622c88a445ca3af6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_36a4391cbd10051042752bbf541ecd6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 4, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_94ab54c0c1b677b8781ba059e1679e56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 4, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c451e9ef5ea5a55eb933aa4d6939e7b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 4, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_00993eca08e040fde6fd176600da0760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_93632e0928214b62aabcc3de219ed808(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_98f63e18dfeefe01a3730694623690ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_46aa33b60f143a70ed0dd739aee53ce8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3024, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5467abcfe71bda4cb341418246304d51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3024, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e491401f13c15ddad765497065d71d9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3024, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e5679fc4917e42d5518a9ac640fe1d21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cb1dd96e658737df955fcb6f9ff8273d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b7c50d0124f467c072a5612140de8f8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dae8a77d56b298259edfb1a6fb838a57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d82168db85b2994d263d64d94bfb37b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fc6f8dd9c7ded518ef7b39dd96fc88c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6bd45aabc3929f6fd902ef0fd70c206a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_172e4fbc521724caf1bd696075e7e83b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8d662f9214d3c1d65116b7a7cb7ffbf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ed0d0554dd72c61f55394edeb16fd035(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_90799ca1f53459a99ebb09d238a17874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_82943a5d17b29a7f9aedd17843256651(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_18b80d7bc57f758418ec15b00e65cb94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f6c9715db32314240d91d94a045a08e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ec405ba4f42332e1acdc2b613149e41a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2fca3db633cd6dbb61c95f7d0852d658(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e18014cf891056ca9f02acd6fcb4539b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_914a9b4daf4417390584e77c38d8f08b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aae7be097487b523f9b24e6de7c4a1e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7714e2b7e0c021107a2745b850821f09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_331bc31cc91fd593bcd339d5346506ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6a9ac92a3bb5d77f7f12bd94bbd48b24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_614d42f90c92bc7b08576f1bd4ba3fc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_db5452b7f4f46ab2aedd3a41b0873bf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd264928eca64f7525dd85147763a5fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5b3292642819d3388ae992b92cd7888a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cb92eadb3fc3a740b8ee9c1e766a1bff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6a2a90587a7b442dd5d0a067bc33b323(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_09b41d4d5a8269e43cf45149bac358dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0123987e2cbdcc72d5227980492b212f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8a72e0f01320df72898f252ed10f8d94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ad091f8bd72b0997bc429b2e440f20fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b5bfce30b14b09fea63572bebc7a8495(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_833cc1fc6836171a089fe6b1ddf0ff55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_df91799a58dae1a72abf0512686e9db7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b65ed4bd2d1848e3ecd364a7e67ae64c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b4bd7c3537ed8a3e79f065cea961329c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3288fb7e4bc4d60afba744084dcfb864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_83fafc0e0dd53365682c69e69ea8569c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f35740555e4074209461b8454b9c0897(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3d7be463e3e8c9591316987e044ac7df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c6104e21dd878246d075f3c5cb31cc3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_40654809da6fa1d508843a4d5995e3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bd04b1ac7fbb5315a0b00efb074a1902(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_57516e0578e80979faaad09105ab28fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1f0d914875c5eddcf57b7b4877c083c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_db835c9926d92aa6d1c6db13a49127bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2f0da3b43d07f95ff29ee1e71e871806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_27b5d6b9ff0fe16cddfe550b492a055b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a602603a8266cc4acb5a7917038f3b93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
            paddle.to_tensor([50], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a08c343128f5a9e98a89439259d178c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([50], dtype='int64').reshape([1]),
            paddle.to_tensor([51], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c5d0f244b897f6a04f683a83fdec017f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([51], dtype='int64').reshape([1]),
            paddle.to_tensor([52], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b9e7c0553177ba68611aa548621bae08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([52], dtype='int64').reshape([1]),
            paddle.to_tensor([53], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bf7e4686c907093843a837be7fdab0d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([53], dtype='int64').reshape([1]),
            paddle.to_tensor([54], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6dddfaaefd235698acf663ea56cf81fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([54], dtype='int64').reshape([1]),
            paddle.to_tensor([55], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c32d761d15b093becaabe1a9b0d4d2dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([55], dtype='int64').reshape([1]),
            paddle.to_tensor([56], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_16695279c1bc5caf21ec803efdbac229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([56], dtype='int64').reshape([1]),
            paddle.to_tensor([57], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_278db4cf9991f49d03e94fbca57e1194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([57], dtype='int64').reshape([1]),
            paddle.to_tensor([58], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_768b628dd29a79985c3406090ccbce8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([58], dtype='int64').reshape([1]),
            paddle.to_tensor([59], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_229a995ea6357db02f28b2fff2214b55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([59], dtype='int64').reshape([1]),
            paddle.to_tensor([60], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_31da2bb29db1e50dcf5b74332807e879(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([60], dtype='int64').reshape([1]),
            paddle.to_tensor([61], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d4b849bc6ef7d6981d9e8e10e9cffa9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([61], dtype='int64').reshape([1]),
            paddle.to_tensor([62], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_08a55b7ee54981e9dfd225890a30d0c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([62], dtype='int64').reshape([1]),
            paddle.to_tensor([63], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b0ddc6afc32d10bb4b647212d68ae39b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([63], dtype='int64').reshape([1]),
            paddle.to_tensor([64], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0aa0abf1eb023bcb7003ff560d57be08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([64], dtype='int64').reshape([1]),
            paddle.to_tensor([65], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a5bc1272143923afad67356be36e68bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([65], dtype='int64').reshape([1]),
            paddle.to_tensor([66], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a1a26688181822b626c5fb4bd2ba96c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([66], dtype='int64').reshape([1]),
            paddle.to_tensor([67], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a718e72db690effb68df2f6adf11bcb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([67], dtype='int64').reshape([1]),
            paddle.to_tensor([68], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6b65486f8a34c032ce0d3e16cc7889f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([68], dtype='int64').reshape([1]),
            paddle.to_tensor([69], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_41f882a132cdfc12795d43da86962ea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([69], dtype='int64').reshape([1]),
            paddle.to_tensor([70], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_086a3872cfd1e2b3192d801f24b88926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([70], dtype='int64').reshape([1]),
            paddle.to_tensor([71], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_26c4a5ced13e064fceefe2359568356f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([71], dtype='int64').reshape([1]),
            paddle.to_tensor([72], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8c1df30448cc5ee07aba3beff21c3012(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([72], dtype='int64').reshape([1]),
            paddle.to_tensor([73], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1a953fc2739acb2cb233dae96b5b43a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([73], dtype='int64').reshape([1]),
            paddle.to_tensor([74], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b1801fc64134e8a1ae871704013499b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([74], dtype='int64').reshape([1]),
            paddle.to_tensor([75], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_32b4d19176d32e19993a7223777530a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([75], dtype='int64').reshape([1]),
            paddle.to_tensor([76], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d78e1d5f0ac83d0a684309ce678f8de3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([76], dtype='int64').reshape([1]),
            paddle.to_tensor([77], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4e62f6b04d445e7326bc7909491f805f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([77], dtype='int64').reshape([1]),
            paddle.to_tensor([78], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dfca56a48edeb3485d3fcd2ef61211a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([78], dtype='int64').reshape([1]),
            paddle.to_tensor([79], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b3375a84ca306f613df60d81349810c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([79], dtype='int64').reshape([1]),
            paddle.to_tensor([80], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3a7f032564f4c7dc769fa2b74763ecb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([80], dtype='int64').reshape([1]),
            paddle.to_tensor([81], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_31c779b2dc37fb483a3b46996dc61c9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([81], dtype='int64').reshape([1]),
            paddle.to_tensor([82], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4979bf36637d0bb0480fc51e1a22ead9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([82], dtype='int64').reshape([1]),
            paddle.to_tensor([83], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_66ae2f8e137196565c0ea578e9f3c138(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([83], dtype='int64').reshape([1]),
            paddle.to_tensor([84], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6966eb7372ed2fa0fe506f8fa189fd1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([84], dtype='int64').reshape([1]),
            paddle.to_tensor([85], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fbd8c7e51028a465a8b6f60bec673413(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([85], dtype='int64').reshape([1]),
            paddle.to_tensor([86], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1791bd3f41ad01e299b2ede4d90b6637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([86], dtype='int64').reshape([1]),
            paddle.to_tensor([87], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0d70cb00b12bad6f147188ab1d9a25d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([87], dtype='int64').reshape([1]),
            paddle.to_tensor([88], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4b8095ca51b78df1d19281dd7a41f6df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([88], dtype='int64').reshape([1]),
            paddle.to_tensor([89], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b461e5c361187ef3be0089f1756f4978(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([89], dtype='int64').reshape([1]),
            paddle.to_tensor([90], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e245aeda90f705df810ec0cddef1ce9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([90], dtype='int64').reshape([1]),
            paddle.to_tensor([91], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e56eebf6c412f2788a6ef5a6e5116291(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([91], dtype='int64').reshape([1]),
            paddle.to_tensor([92], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9beffc30b47be7bbbf535d920a97764e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([92], dtype='int64').reshape([1]),
            paddle.to_tensor([93], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bd8ce04e7a70f8a395001e7874029bd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([93], dtype='int64').reshape([1]),
            paddle.to_tensor([94], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_14e605e91c7ccee8ace54dc389e3d17e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([94], dtype='int64').reshape([1]),
            paddle.to_tensor([95], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9b72194ce0e29a83550cee20db0cde64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([95], dtype='int64').reshape([1]),
            paddle.to_tensor([96], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_03ae9fb082c714b9a9c5820594ac3ff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([96], dtype='int64').reshape([1]),
            paddle.to_tensor([97], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a9f47b992084a678204ff457e0bf9e20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([97], dtype='int64').reshape([1]),
            paddle.to_tensor([98], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5a9f3f444937ce5688681afa7eb76971(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([98], dtype='int64').reshape([1]),
            paddle.to_tensor([99], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1c7a9a0185300e4026ed40eb0b9e92dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([99], dtype='int64').reshape([1]),
            paddle.to_tensor([100], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ebaa2109f4e7cd27bb607c43636a7e63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([100], dtype='int64').reshape([1]),
            paddle.to_tensor([101], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_97ccb75ae77318b8eedcbdf6c838b220(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([101], dtype='int64').reshape([1]),
            paddle.to_tensor([102], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2ad88ec7444c4db11676cbf06f87721c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([102], dtype='int64').reshape([1]),
            paddle.to_tensor([103], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f55e410bb3a379815a2cc766422a361d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([103], dtype='int64').reshape([1]),
            paddle.to_tensor([104], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e3366ff57c5537b6ef0ea426fe04533(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([104], dtype='int64').reshape([1]),
            paddle.to_tensor([105], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_064d8a0c24069a4cd9fcf91ecb2547f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([105], dtype='int64').reshape([1]),
            paddle.to_tensor([106], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ea2175716e8aae3e467eb6de1eba506b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([106], dtype='int64').reshape([1]),
            paddle.to_tensor([107], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c6e6fd44e1250ef4030c82a1cb3f0066(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([107], dtype='int64').reshape([1]),
            paddle.to_tensor([108], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ff6c476862b334b35146b0f49100f715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([108], dtype='int64').reshape([1]),
            paddle.to_tensor([109], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e9be9abf92ec7b8e8bec0ac8ea1ee867(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([109], dtype='int64').reshape([1]),
            paddle.to_tensor([110], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_82e1006065d9889577ca70b9eb14b49f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([110], dtype='int64').reshape([1]),
            paddle.to_tensor([111], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_180f5d2990c994cdaaf5710833a6de56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([111], dtype='int64').reshape([1]),
            paddle.to_tensor([112], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ef9bd9114e2a9257df4c348173ab97a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([112], dtype='int64').reshape([1]),
            paddle.to_tensor([113], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_01fa7b6410d4d4f3089e59c35e077353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([113], dtype='int64').reshape([1]),
            paddle.to_tensor([114], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_abcb2372e6f811b3c99a1250b3364cd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([114], dtype='int64').reshape([1]),
            paddle.to_tensor([115], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_56c1731c4f575fca91cdf94460188753(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([115], dtype='int64').reshape([1]),
            paddle.to_tensor([116], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86ee73f0d40168f851e15fb0b17ed737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([116], dtype='int64').reshape([1]),
            paddle.to_tensor([117], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_627761b58af812ef483e26dc66f48e62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([117], dtype='int64').reshape([1]),
            paddle.to_tensor([118], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9fe8ee202d0584a8cffe560f09937c20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([118], dtype='int64').reshape([1]),
            paddle.to_tensor([119], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_444b6ae8d70626776fa9a394c64ded31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([119], dtype='int64').reshape([1]),
            paddle.to_tensor([120], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3a8078ed9b50692e5ff7128156b617c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([120], dtype='int64').reshape([1]),
            paddle.to_tensor([121], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5a9b9bdf1fa7d3a72f56d2d47f9eea9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([121], dtype='int64').reshape([1]),
            paddle.to_tensor([122], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1436d37bfc93a4028b661d5701a5cedd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([122], dtype='int64').reshape([1]),
            paddle.to_tensor([123], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7fc0b29201bd082f2c43969c4eff9728(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([123], dtype='int64').reshape([1]),
            paddle.to_tensor([124], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9f783a9fd9c791709acea63234cbf4b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([124], dtype='int64').reshape([1]),
            paddle.to_tensor([125], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_596a46243011914919a61ce6dc3f08d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([125], dtype='int64').reshape([1]),
            paddle.to_tensor([126], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8fc2f153ecb2ed79c9a282c59cbb898e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([126], dtype='int64').reshape([1]),
            paddle.to_tensor([127], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3e31d5035f4094cecc11500a088bd868(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([127], dtype='int64').reshape([1]),
            paddle.to_tensor([128], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fb7d3a230fd34facfa72b93e3eaeb547(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([128], dtype='int64').reshape([1]),
            paddle.to_tensor([129], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47f13d0c22869580d3bcc70325ebf99d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([129], dtype='int64').reshape([1]),
            paddle.to_tensor([130], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c6aac6947612034ebc09054dd7b5defa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([130], dtype='int64').reshape([1]),
            paddle.to_tensor([131], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3d36d43fd445e0ba8dd1ec0ae59205f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([131], dtype='int64').reshape([1]),
            paddle.to_tensor([132], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d33f01a6ffd28253028b6a865c9d5281(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([132], dtype='int64').reshape([1]),
            paddle.to_tensor([133], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_836aeddd4f5a156d9dc515e348b6a113(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([133], dtype='int64').reshape([1]),
            paddle.to_tensor([134], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_65451a5a1914ae1ed63d05df2e5ac9e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([134], dtype='int64').reshape([1]),
            paddle.to_tensor([135], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_79b3a0bde4fe54b2c71c5f96ac7b637e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([135], dtype='int64').reshape([1]),
            paddle.to_tensor([136], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_10b17d6377971716daa40408a494b079(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([136], dtype='int64').reshape([1]),
            paddle.to_tensor([137], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_72ed302ca76261af0df811c9559d07b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([137], dtype='int64').reshape([1]),
            paddle.to_tensor([138], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c2278e5063757061283b448a9bf3c55f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([138], dtype='int64').reshape([1]),
            paddle.to_tensor([139], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eed32d37fde54b5e2b9f985ea646e26d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([139], dtype='int64').reshape([1]),
            paddle.to_tensor([140], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1fc3f5b3d3dbea4ea252c96d1a21e374(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([140], dtype='int64').reshape([1]),
            paddle.to_tensor([141], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0a84ec4a5cbe82916781b40d9a56c736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([141], dtype='int64').reshape([1]),
            paddle.to_tensor([142], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_399917605fb71e4f9e45cddf13e898b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([142], dtype='int64').reshape([1]),
            paddle.to_tensor([143], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bf63b0f8a398162c69661f7dc99abeb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([143], dtype='int64').reshape([1]),
            paddle.to_tensor([144], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_abea662f1fd1a82ffa205ca68d7771f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([144], dtype='int64').reshape([1]),
            paddle.to_tensor([145], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf0436b828861378060ffb4f630a9938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([145], dtype='int64').reshape([1]),
            paddle.to_tensor([146], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ff4188c4b00cbd1950a12682e00bdc10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([146], dtype='int64').reshape([1]),
            paddle.to_tensor([147], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f2ddab055f7da36396b5f11e9fa38916(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([147], dtype='int64').reshape([1]),
            paddle.to_tensor([148], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_09696296e479f8eac9547c50bc148ac7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([148], dtype='int64').reshape([1]),
            paddle.to_tensor([149], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_30bd08e5a495caa1778248b8eaf0bf8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([149], dtype='int64').reshape([1]),
            paddle.to_tensor([150], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5e1f07fa2e342cba6544d2fbd1c4e73d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([150], dtype='int64').reshape([1]),
            paddle.to_tensor([151], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b96fea84a247d438abc74ea445e45e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([151], dtype='int64').reshape([1]),
            paddle.to_tensor([152], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c2ae5153eb462e398d95075acc4fe691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([152], dtype='int64').reshape([1]),
            paddle.to_tensor([153], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7f581ce27b7738d6609659d2d5a1784a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([153], dtype='int64').reshape([1]),
            paddle.to_tensor([154], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e5c2013b23d4bcd954b6e9c20e7e77a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([154], dtype='int64').reshape([1]),
            paddle.to_tensor([155], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a07f046b0b5215a5f7c103c2ca0026a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([155], dtype='int64').reshape([1]),
            paddle.to_tensor([156], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3b6e37833660cec5abefe02c412f0ec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([156], dtype='int64').reshape([1]),
            paddle.to_tensor([157], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b37fdfb3fdcb1ae3a49f358ce124c161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([157], dtype='int64').reshape([1]),
            paddle.to_tensor([158], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d84727839821bf7d88296584c03f37ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([158], dtype='int64').reshape([1]),
            paddle.to_tensor([159], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3471dc067267a6f057d0d264ab5a0403(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([159], dtype='int64').reshape([1]),
            paddle.to_tensor([160], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c4b5ed953b9035c7a808c9da490036d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([160], dtype='int64').reshape([1]),
            paddle.to_tensor([161], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a66d4d77fc997083efa391d6aa10dda3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([161], dtype='int64').reshape([1]),
            paddle.to_tensor([162], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_55da92338856fc12b89c8b0953db35ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([162], dtype='int64').reshape([1]),
            paddle.to_tensor([163], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_24b34ea21d4715cc714fdf922ef8a9f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([163], dtype='int64').reshape([1]),
            paddle.to_tensor([164], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_62e514389ed99cf6a7cb1c18b00efd0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([164], dtype='int64').reshape([1]),
            paddle.to_tensor([165], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8d83fa80b4084dfea592d675f66836c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([165], dtype='int64').reshape([1]),
            paddle.to_tensor([166], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ab5fa0864c6eeb1661bbe29c51af05fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([166], dtype='int64').reshape([1]),
            paddle.to_tensor([167], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_217f210e0b0731a069fba6dc6be3a815(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([167], dtype='int64').reshape([1]),
            paddle.to_tensor([168], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5da57287996da7964fcf06b15e665a49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([168], dtype='int64').reshape([1]),
            paddle.to_tensor([169], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2e28b2c88d7f5daeb07d8f289333d2b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([169], dtype='int64').reshape([1]),
            paddle.to_tensor([170], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84aa2885969b6c93db8878af274c90c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([170], dtype='int64').reshape([1]),
            paddle.to_tensor([171], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0e17c603fb9719fa346b403a8e766603(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([171], dtype='int64').reshape([1]),
            paddle.to_tensor([172], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f242f0d97acda79df8decda3c654d855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([172], dtype='int64').reshape([1]),
            paddle.to_tensor([173], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_608d9291174bc0d3a95b646c870f6178(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([173], dtype='int64').reshape([1]),
            paddle.to_tensor([174], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_70679593cb542045d4062ce24b4b486b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([174], dtype='int64').reshape([1]),
            paddle.to_tensor([175], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b4acc5df943a569a34065e39db1b8421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([175], dtype='int64').reshape([1]),
            paddle.to_tensor([176], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_00c1abf241396cc84282ba7de604c5a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([176], dtype='int64').reshape([1]),
            paddle.to_tensor([177], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2dfb2a150c2db01dd6d9293cebe2fb8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([177], dtype='int64').reshape([1]),
            paddle.to_tensor([178], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f13ea856c4f5340b08b1e351cd43aae3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([178], dtype='int64').reshape([1]),
            paddle.to_tensor([179], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0db3f95ba8c3dfce14d86c472593db3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([179], dtype='int64').reshape([1]),
            paddle.to_tensor([180], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ea9ef718e57cc62ce693b15909793580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([180], dtype='int64').reshape([1]),
            paddle.to_tensor([181], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b7b4c9b008e862cb1b1065f8cfd69955(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([181], dtype='int64').reshape([1]),
            paddle.to_tensor([182], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_130e2cc5c59985dc9e26d6c985501267(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([182], dtype='int64').reshape([1]),
            paddle.to_tensor([183], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6cd5678f19d1adebcdd9c0a2ee007535(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([183], dtype='int64').reshape([1]),
            paddle.to_tensor([184], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_36bc299854ed099928e56f9f9d7e5827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([184], dtype='int64').reshape([1]),
            paddle.to_tensor([185], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_381d39c10e79d7ba598e920747f98d42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([185], dtype='int64').reshape([1]),
            paddle.to_tensor([186], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_72b44f2b69975d77c9e080e935109f37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([186], dtype='int64').reshape([1]),
            paddle.to_tensor([187], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b5b516cdd8164ed20ed6245d1d597e42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([187], dtype='int64').reshape([1]),
            paddle.to_tensor([188], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3cf2e59b77c4608b108fcfe0c8638ba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([188], dtype='int64').reshape([1]),
            paddle.to_tensor([189], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0d0026f254e602cad680eee71b18e682(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([189], dtype='int64').reshape([1]),
            paddle.to_tensor([190], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c827015a717708d68f4731ced04adea3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([190], dtype='int64').reshape([1]),
            paddle.to_tensor([191], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7adc553ea1a2241ef76f365c3530e4f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([191], dtype='int64').reshape([1]),
            paddle.to_tensor([192], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_579c1f7a93a73c098afdc9421b2f7470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([192], dtype='int64').reshape([1]),
            paddle.to_tensor([193], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6eb290333bb218fd466a1c822c210132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([193], dtype='int64').reshape([1]),
            paddle.to_tensor([194], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a31956d340a92ac20868a75ccd482389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([194], dtype='int64').reshape([1]),
            paddle.to_tensor([195], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4ccb222a9ec48180cda5cad82e040cea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([195], dtype='int64').reshape([1]),
            paddle.to_tensor([196], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7e4e6cef4535de2ba828ccda03345ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f1513860488fd669a3da36e20551cad4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a44edf436e5b2c2ebf1b8850d8288bbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f7dcd1b81bdecd7093ed818423d1244b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c1ab3cc9bd3bdb87ce0e5f589011bee3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f6d6fc49feee5712c9d98a39b2d2d332(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_73b571a8db56a757dd0900fd37157fdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e6692cf6b4aa5f29db24ad21a08eea7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_016edbf8e8a35903dbcb385a6dd79317(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3eecc25f9889873841ae6779d472cba7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8dc27ba0bdcbefac0049081a37661348(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a1787d8313ba2aeb6529f8c70d492d84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_099bb9e3e1821ec6ef79ff376b9ad071(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4a5b606f46fdb15f88703e70a118639d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9cc19119ebf563d60fc918a0ed550c5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b0284685276bbf37929ce806bd8c01fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7887074041de7c51b90ea53d176499ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_25df3010585bc230fefb8584ef8c01cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1f9c0ab5f7135b1818e22ef62fcf7455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_64cccaac64581ea33257337c3578e619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b99710df460049d687837b1f7128b7ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_82d70d6a3c1cfdd52198edd8a8d83ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a0f99f6b16b53e5b9cc41f4e53036811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dc69a94d6fcef9b1623f1634ee47b57c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8caf831903892b93f1ff80c4e7763257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f66c96377def52417466d9d2c428a2f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bcb5e88cf860004dea3e6a2e65fb5483(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fed9891096e5394080264560fc2241a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1d86e2ea7b2d02bc5d5d2ba7fa6b8e0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c18875264a6a541950d032cf4db92174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3f1b49ed9e1225b04e11f3b89cde3bd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b67f431a8d6433ab3a18e2c34b4d221c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_06cfcf1a880b3d2689f8c7e5843ba7da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0d1fc242d9608879213ee1de49733111(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_500a8a0a67eef18d8f19f0923953a2c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f455b3b7e551d7a1cb9d9f11b85a8ae0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_22994c09f40d290621fbb24d6d37eb15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5ee3531250cdcc51d9ff0f96a5e0182f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b30b15905785f4c44b45d51f918b4806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7a38002c640d11fbd0071e1b36e4b9b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b7daa3745ed6b9c1ec560d824a8f59e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b38e78c9732fc507ad55bdb08f15243a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0e1d8f473505590c7a302c3d24610640(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_897dbc98f01f791b2fbb7cdf4ee270d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f6b9bebc0a74f93ee7ac04badae677aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_045d2d24b1dee6bb25c9b3aad99f42b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_33f483be79e8f4f2bc693ebca3554a2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bdc58f70bf2227e610b855927c878d4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_616a0369de42dcc033c1719061174357(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bb276e7494ed1fd044843492e5203f37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eaea493d6e0419e82f2a4f7da6610949(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_78e771f1e275a54fe6005a7821d2b94b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a472edeb627d9edfb5a0ced01ef797a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ecfa67c5ecf10a7b0268c9dc72a9062f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d29267296303f1cb6c91b2600bf4d275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ef8b9513887f684f1ae605f3bee79b25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4b91df9b18a6aef5697b37458740bfb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_112abcb5a2c6d2d53a54a7db6f96a5b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a906a93dc0b02558ce9fe468b60a69bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_61e5d729a672a247f15fe36e62706fdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ee64b1e7c09c4d5af181448e2b1003aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_19af38d77dbf3ad35e0aa8951abf832c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1ccc7f9e28fc2ce4c05b31d97139bb81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_778b1c4f4af4211a94dde93143e032a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d03227d7d2ef89bbd9392d929437a256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_72e37498179852539d5fa22019f2a7f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dad39b53169e2fcc6205f03948e51274(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5955756c9125fd0fc4fa43156a3d10b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a7fa586ae347193f38332e2a55037b23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1174, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_662ea8c8c51dbe64a414bb461f0ba9fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1174, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8329cd8c50a8795a28c73a6973cdc928(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_13c02dfb7d1e28d7c8b8e0b3c6b64f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_43f29c6a37dd1915e70786aede05dacc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ebde116a205f47f8a66622e34485a7c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f222e4d3bba2017da376e8753189c87d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_52f9c18af0dd5dd922dcdcc434ee108d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e0b07a6edc7b8d38e5bc555dbd6f338e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bdd5954db65257c0a0487268ebcbf09f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15d3433429adf42d48390a15903c16dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cd6277ee13835399ac92102d0eb5c6e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1d04419020b6b6ccbcb3c7cfe2adc0a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 116, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f2a3056376c35d87f17cb1ff0af0f18d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 116, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d55e24701d8851c26d450c7a58db93ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 116, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e9a74e878289dede85d1a2af4b34c3dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 116, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bac95db01bd1694f2888008d957ff19b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1174, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d7feccae6098089fa1f60727ff6d68e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1174, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cd008588568746c442b215827ecddd4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5e22b53ab2e110c35ce1a25f5c78ccd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9d8d16eec12e97228c533f3d902ba2e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_434ec48cfa0e682a794b7cd135a4e92e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_48b1c6edaf7ff9c0936485ae231707de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5905642d885065b305c32411cd278349(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d83f15b31cc23bda8f173b53946439c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c95850ace86ccc62a4cf1925a0ff8bf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c2eba266db7c82567621f76b113b595d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_106b2a337eb1a41ac706c15f82438f22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c86f0ecf2722781fb1daead6bd6367ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4f1c034e55f3b64adc5df814e9d31d42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_af31f553c24d970c8b22e9a3eb98eb07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_51f6d0ed845c43c05e4c7252cb1c0ed0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2519dd245a3fcd365805857f7a80af90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0802051957311baba930abf587377eb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9dd9c1a50be1201824c0e710335fa928(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_13595eea850c403714e92ce1f39a915a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6d6a57d7d21b088ce43cb6729bb18e7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f9f8ed10c5b7e7a3a572b8b99521dfcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b6f1bcd7073b568b5f7213e2c0ad7c2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5af67120d7ba08f0b11a90fb8721050a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4729bdbae2bdabf3bdf64c068c61d729(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()