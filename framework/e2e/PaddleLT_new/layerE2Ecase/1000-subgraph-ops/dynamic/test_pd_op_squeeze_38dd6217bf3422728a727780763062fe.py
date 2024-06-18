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



class PrimitiveOp_05e3fa6ae2d5447bd2ce2ab9fcd2eae1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8a1d88e4b3747389963191fb99331a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05e3fa6ae2d5447bd2ce2ab9fcd2eae1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[15200, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39dad310b143ada1869fb8ff587aac91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([1524, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_39dad310b143ada1869fb8ff587aac91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([1524, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5b073cda074a1cffd2d9fb0e1311afd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([2340, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5b073cda074a1cffd2d9fb0e1311afd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([2340, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_93ffbaf2e5d774a2145ef98bc02dbc90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05e3fa6ae2d5447bd2ce2ab9fcd2eae1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3800, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_93ffbaf2e5d774a2145ef98bc02dbc90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05e3fa6ae2d5447bd2ce2ab9fcd2eae1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3800, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15d490f828f2d29d5744c42a8895515b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05e3fa6ae2d5447bd2ce2ab9fcd2eae1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2204, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_87d0e3fa4541252d9c6fc0f8484b4755(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([2047, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_87d0e3fa4541252d9c6fc0f8484b4755(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([2047, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b804a63fffd323d31250af8dfe461d8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05e3fa6ae2d5447bd2ce2ab9fcd2eae1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[950, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e8a1d88e4b3747389963191fb99331a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05e3fa6ae2d5447bd2ce2ab9fcd2eae1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[15200, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_873d5c5421bfb92b9c0337306d05fd9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([1813, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_873d5c5421bfb92b9c0337306d05fd9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([1813, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7425b57584bea0d262c3eb3eb4ce71a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([3061, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7425b57584bea0d262c3eb3eb4ce71a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([3061, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0fa406158e112b92d3853c6c54c15e5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05e3fa6ae2d5447bd2ce2ab9fcd2eae1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[8816, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bf68221baeef7fdab008a8f7465c5961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([2062, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bf68221baeef7fdab008a8f7465c5961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([2062, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6d747e9f9afad73b29f77ed15e1bcb0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05e3fa6ae2d5447bd2ce2ab9fcd2eae1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[150, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_952e1f5281c8073ac8cb73514b86e13c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05e3fa6ae2d5447bd2ce2ab9fcd2eae1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[70, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8498a9c1924774a9f72f737435ced4b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([5526, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8498a9c1924774a9f72f737435ced4b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([5526, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f14ac650968d4765cd969bd866d5ecf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([1071, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f14ac650968d4765cd969bd866d5ecf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([1071, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ad92bc04683c697a979bef2eeb8bde20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([1760, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ad92bc04683c697a979bef2eeb8bde20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([1760, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b804a63fffd323d31250af8dfe461d8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05e3fa6ae2d5447bd2ce2ab9fcd2eae1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[950, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_952e1f5281c8073ac8cb73514b86e13c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05e3fa6ae2d5447bd2ce2ab9fcd2eae1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[70, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0d48d9485e01c938dc9110e370f6f3ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([4204, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0d48d9485e01c938dc9110e370f6f3ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([4204, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a2a921d26114967092bb70cbf8b5f95c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05e3fa6ae2d5447bd2ce2ab9fcd2eae1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[551, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_93ffbaf2e5d774a2145ef98bc02dbc90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05e3fa6ae2d5447bd2ce2ab9fcd2eae1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3800, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1fae205c7d0b10276fc7af311242066c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([4680, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1fae205c7d0b10276fc7af311242066c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([4680, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1da160e9aa376bcd3e3463560f3d179a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([3778, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1da160e9aa376bcd3e3463560f3d179a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee820e9870d60a10c8f255a2b7b5dea
    def get_inputs(self):
        return [
            paddle.uniform([3778, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a8266027a04ee500e23fd1810b70fb26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05e3fa6ae2d5447bd2ce2ab9fcd2eae1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[247, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a8266027a04ee500e23fd1810b70fb26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05e3fa6ae2d5447bd2ce2ab9fcd2eae1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[247, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()