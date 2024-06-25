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



class PrimitiveOp_9658652e70a6c6f545177c3dcc4d9458(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 1]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e11f1b28f0f553b00456582cf4b4495(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9658652e70a6c6f545177c3dcc4d9458
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 1], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_01dba6b5a94e33cb8cf9011e5a1a3df0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 4]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_225b6e128ebc1ffa325f2403ed8f0fb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01dba6b5a94e33cb8cf9011e5a1a3df0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_1bf07e7481d778207c9b845996e351b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 68]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dccd1d3e6cfa5eb31e83d80b52407338(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf07e7481d778207c9b845996e351b7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_291eb0a525fef650bfcc699656f0b492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01dba6b5a94e33cb8cf9011e5a1a3df0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 11109, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_d2db40928055fd2d4c50dac25487b3a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf07e7481d778207c9b845996e351b7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 11109, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_225b6e128ebc1ffa325f2403ed8f0fb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01dba6b5a94e33cb8cf9011e5a1a3df0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_276f7fbced0866081999776b9b7eabe3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 76]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c6c44a3fc5b1d7630d4371a0b5d695b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_276f7fbced0866081999776b9b7eabe3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 76], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_7c50b8f3ed41fa1b86e5e93e0d60945c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01dba6b5a94e33cb8cf9011e5a1a3df0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3024, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_4d8398e42c85fad6ef78a41cfc6d8d5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf07e7481d778207c9b845996e351b7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3024, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_7c0485dfa06b6945d4df6a01a1539e7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01dba6b5a94e33cb8cf9011e5a1a3df0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_10af7b8fb0d9ad6e2a85b2847f2a10de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf07e7481d778207c9b845996e351b7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5f03714c4030dcacb612826f44eaf1c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01dba6b5a94e33cb8cf9011e5a1a3df0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 9261, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_2baff2a8ce87fd59fcc9903c05ca5369(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf07e7481d778207c9b845996e351b7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 9261, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_e7fb4b2a48a0697f8447ada0ffd76897(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01dba6b5a94e33cb8cf9011e5a1a3df0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_0964addf1eeb1e1c983f47b3e7753b5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf07e7481d778207c9b845996e351b7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_2c1909a3d2ad5316ea4cb8933c301609(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 100, 1]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c3b3056734f4c37080890058aaf67dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c1909a3d2ad5316ea4cb8933c301609
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.4188864231109619, 0.4517028331756592, 0.15065787732601166, 0.030897535383701324]]], dtype='float32').reshape([1, 1, 4]),
            paddle.to_tensor([1, 100, 1], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_901a01de81e498de45e5ad9630ca2e30(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 300, 1]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a0d5ba1830db50a48b400c04ae589fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_901a01de81e498de45e5ad9630ca2e30
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.4614632725715637, 0.3346860110759735, 0.0953991562128067, 0.0004730565124191344]]], dtype='float32').reshape([1, 1, 4]),
            paddle.to_tensor([1, 300, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_aec1c96b59fe848bebc11cb5bf977b56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01dba6b5a94e33cb8cf9011e5a1a3df0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4725, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_971bc28a1b266ae71ed90b4c58d2a2f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf07e7481d778207c9b845996e351b7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4725, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_dba72c6384dbd00357b8044d24af97f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01dba6b5a94e33cb8cf9011e5a1a3df0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 6069, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_4b3ec635017ce0201d7c71c85c9c5e6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf07e7481d778207c9b845996e351b7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 6069, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_e0c24c67a3f5d9a606b62caaaedb0d06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01dba6b5a94e33cb8cf9011e5a1a3df0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 7581, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_ff368f6fb09b9779dc1ef5d0e3090b24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf07e7481d778207c9b845996e351b7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 7581, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_bd69b68a1250a0d9fddeb2d34f0c565c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 512]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02d3ace38cee22615d2813eb7ef4c6e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd69b68a1250a0d9fddeb2d34f0c565c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 512], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_7c0485dfa06b6945d4df6a01a1539e7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01dba6b5a94e33cb8cf9011e5a1a3df0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_10af7b8fb0d9ad6e2a85b2847f2a10de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf07e7481d778207c9b845996e351b7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_02d3ace38cee22615d2813eb7ef4c6e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd69b68a1250a0d9fddeb2d34f0c565c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 512], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_3cc3a300e36d78058bfd4a392fac6dcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01dba6b5a94e33cb8cf9011e5a1a3df0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8400, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_3ab6f421f57835ba8663652723d12839(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf07e7481d778207c9b845996e351b7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8400, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]




if __name__ == '__main__':
    unittest.main()