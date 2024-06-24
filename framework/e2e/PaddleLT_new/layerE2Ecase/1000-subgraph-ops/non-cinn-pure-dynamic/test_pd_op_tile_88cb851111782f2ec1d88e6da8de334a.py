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



class PrimitiveOp_80ab0885dd242442f92a7d720e75d3bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_15dd53bf752383a3244ec533566c6ece(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ab0885dd242442f92a7d720e75d3bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 1], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c8db04cd344e5a1bab60d42d3241e7ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_68da966ae79722c933259701827caaa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_dddb625553882e293b854e2eb1f23f56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 11109, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_6de8dad7e5c3d3dbb57ac2d58f3d6ea2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 11109, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_c8db04cd344e5a1bab60d42d3241e7ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_39cbaeb38e0acefae2934ec46f369a0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 76], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_383d19ebcaf20fe836c15ccd6324e4df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3024, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_dae6617ca6ec69549e872802a6ca4e82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3024, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_9e2a20a155ad80ee397ab45ff230193f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_743d6e215cd9c7ba594b42eb45484f45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_6bde1933aa675e5a41cea5dad6edd4a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 9261, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_23b6473c5e529e135a01dc573f0f3bbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 9261, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_d8026ed42ac5e0250a0812b36eef78c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b0458ca6af2b6d3df1322a9cd3150a82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b0449f72899475e0e55cfda1b52cbfc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ab0885dd242442f92a7d720e75d3bc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.4062451422214508, 0.008214853703975677, 0.2228037714958191, 0.2746564447879791]]], dtype='float32').reshape([1, 1, 4]),
            paddle.to_tensor([1, 100, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_52dabd77143aaa72f189780aa6de888c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ab0885dd242442f92a7d720e75d3bc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.06691982597112656, 0.2933773100376129, 0.1433459371328354, 0.30485281348228455]]], dtype='float32').reshape([1, 1, 4]),
            paddle.to_tensor([1, 300, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_9fdc6f9bdf231b49280b9d774c80066e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4725, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_0a72b19dd59a4165afca3805b28cfa5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4725, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_c12ac0e7e83dc8214e728c7f9439094e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 6069, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_9d027afc0a083dbe7cb9e37522f9a62e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 6069, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_8c69cfeab4d2c7e5f716eb22a0a1b1b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 7581, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_31b11e2f64cdbf6942630ef328285323(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 7581, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_ae7df851c084bb4088bbbf7be1a0fe15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ab0885dd242442f92a7d720e75d3bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 512], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_9e2a20a155ad80ee397ab45ff230193f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_743d6e215cd9c7ba594b42eb45484f45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_ae7df851c084bb4088bbbf7be1a0fe15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ab0885dd242442f92a7d720e75d3bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 512], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b9dfac80403cdbeb41b16990d8c3588b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8400, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_53055769c09d7981c571f04da5e0f403(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8400, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]




if __name__ == '__main__':
    unittest.main()