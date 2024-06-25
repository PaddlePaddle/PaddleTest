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



class PrimitiveOp_1aa906736f7211599b508766917ff1dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1024, 1]
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74ae8812b82fd96a485492339494c323(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1aa906736f7211599b508766917ff1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 1024, 1], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_0fe34dae687d5d8e3839e32ae8175565(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 4096, 1]
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f712fc0496fb0e4d42b2c765a65a3dea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fe34dae687d5d8e3839e32ae8175565
    def get_inputs(self):
        return [
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 4096, 1], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_bafc4f85714da4ca72e8cfdef8febdc7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 16384, 1]
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_847a4112648050f6926e481db1be55bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bafc4f85714da4ca72e8cfdef8febdc7
    def get_inputs(self):
        return [
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 16384, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_74ae8812b82fd96a485492339494c323(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1aa906736f7211599b508766917ff1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 1024, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f712fc0496fb0e4d42b2c765a65a3dea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fe34dae687d5d8e3839e32ae8175565
    def get_inputs(self):
        return [
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 4096, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_847a4112648050f6926e481db1be55bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bafc4f85714da4ca72e8cfdef8febdc7
    def get_inputs(self):
        return [
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 16384, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_847a4112648050f6926e481db1be55bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bafc4f85714da4ca72e8cfdef8febdc7
    def get_inputs(self):
        return [
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 16384, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f712fc0496fb0e4d42b2c765a65a3dea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fe34dae687d5d8e3839e32ae8175565
    def get_inputs(self):
        return [
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 4096, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_74ae8812b82fd96a485492339494c323(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1aa906736f7211599b508766917ff1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 1024, 1], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_9f5373f812e0ce5d5a73d12f4941d047(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 256, 1]
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9803e7983a5c8a525ddc27bc8bc2e737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f5373f812e0ce5d5a73d12f4941d047
    def get_inputs(self):
        return [
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 256, 1], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_2a78d468a2ca3483dfc3d1df33bf7ad6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 64, 1]
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bae72d5cace7b4260ddea9434eb14319(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a78d468a2ca3483dfc3d1df33bf7ad6
    def get_inputs(self):
        return [
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 64, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_74ae8812b82fd96a485492339494c323(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1aa906736f7211599b508766917ff1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 1024, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f712fc0496fb0e4d42b2c765a65a3dea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fe34dae687d5d8e3839e32ae8175565
    def get_inputs(self):
        return [
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 4096, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_847a4112648050f6926e481db1be55bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bafc4f85714da4ca72e8cfdef8febdc7
    def get_inputs(self):
        return [
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 16384, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_74ae8812b82fd96a485492339494c323(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1aa906736f7211599b508766917ff1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 1024, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f712fc0496fb0e4d42b2c765a65a3dea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fe34dae687d5d8e3839e32ae8175565
    def get_inputs(self):
        return [
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 4096, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_847a4112648050f6926e481db1be55bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bafc4f85714da4ca72e8cfdef8febdc7
    def get_inputs(self):
        return [
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 16384, 1], dtype='int64').reshape([3]),
        ]




if __name__ == '__main__':
    unittest.main()