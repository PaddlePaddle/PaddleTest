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



class PrimitiveOp_e9300524861fd4309823753e1c54cdce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b8b1d1f4ca7a6fe4ad9a2e98f7a98675(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9300524861fd4309823753e1c54cdce
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.38068312406539917, -0.25771480798721313, 0.42517513036727905, 0.4099397659301758]]], dtype='float32').reshape([1, 1, 4]),
            paddle.to_tensor([1, 100, 1], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_74bdf04379c2cf08b1890ffc93a8b780(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d864682b272223b35f9e8de6a7c9b062(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74bdf04379c2cf08b1890ffc93a8b780
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3024, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_58f370534d17b8b9ab2d173f458b40e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74bdf04379c2cf08b1890ffc93a8b780
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3024, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_39070aec37183ace71bfa1509d87cd42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a0b9a4853fff697548d2e96a9ddfea16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39070aec37183ace71bfa1509d87cd42
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4725, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_07f77f8a57d998242d791d9a8f5e5f6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39070aec37183ace71bfa1509d87cd42
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4725, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5a96b2103dd193b46e499431589228ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9300524861fd4309823753e1c54cdce
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.2312726378440857, 0.08366900682449341, 0.21712779998779297, -0.029599010944366455]]], dtype='float32').reshape([1, 1, 4]),
            paddle.to_tensor([1, 300, 1], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_f7089a40610cc8cfa1e8e2343afc941c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_779591c593448be45614f6c13957d1e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7089a40610cc8cfa1e8e2343afc941c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 512], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_c5e81d8f05e338f8d08497c4e0177ae9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 300, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dd7ff12d5af8194c032d2b06ea6556c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5e81d8f05e338f8d08497c4e0177ae9
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 1], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_cc838d5301fad3429022b3ff54fc0c04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7aa2502b5416fbdd53fea7e4be8bab4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc838d5301fad3429022b3ff54fc0c04
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_2a8fd7a6649fc3494b9bf3a16bb88121(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc838d5301fad3429022b3ff54fc0c04
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_cfb64abd0242436b6b667b9091e633a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_389c9e4bc86a7fd66b6cf1baf23df5ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfb64abd0242436b6b667b9091e633a3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5dbf89bccfef78d0cd351a2f02a72fa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfb64abd0242436b6b667b9091e633a3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 76], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_d7b6cef7c8c4e018183320442fa2d79c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4b00e4327c2e40404fbe8efe9b826f6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7b6cef7c8c4e018183320442fa2d79c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 6069, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_62dbda0ca184a2aa9f80793f2be7ab98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7b6cef7c8c4e018183320442fa2d79c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 6069, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_7aa2502b5416fbdd53fea7e4be8bab4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc838d5301fad3429022b3ff54fc0c04
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_2a8fd7a6649fc3494b9bf3a16bb88121(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc838d5301fad3429022b3ff54fc0c04
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_04c66ea6812eb9c7a1878bf1f46ce833(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_64d891cb9368518091aed374cb75e6ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c66ea6812eb9c7a1878bf1f46ce833
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 11109, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_2763425b18957b09e49313528a6bbbc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c66ea6812eb9c7a1878bf1f46ce833
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 11109, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_4a71c2d70fb8dfa278d2f2f69ef77c8a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ebacf4000dffe22d1b2b6bb80fb189bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a71c2d70fb8dfa278d2f2f69ef77c8a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 2100, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5f825291ffdfc60251cdb46aff403c14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a71c2d70fb8dfa278d2f2f69ef77c8a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 2100, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_389c9e4bc86a7fd66b6cf1baf23df5ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfb64abd0242436b6b667b9091e633a3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_fc8c33b7aafc23a6fdebdef6b430959e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfb64abd0242436b6b667b9091e633a3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_779591c593448be45614f6c13957d1e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7089a40610cc8cfa1e8e2343afc941c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 512], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_0a4736c00227bbd2c8773a45292eeafb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_986b584579eefa1da09ca6cb2322dadd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a4736c00227bbd2c8773a45292eeafb
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 8400, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_2be3f591705eb660c9c261acc3e1b018(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a4736c00227bbd2c8773a45292eeafb
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 8400, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_ef92868966eb8fd907b73da8050e01f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef58f0db1bf133e748cd8868991464b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef92868966eb8fd907b73da8050e01f3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 9261, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_833769940ba88145e89f318834896b47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef92868966eb8fd907b73da8050e01f3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 9261, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_f383b3e6c8a02917f82e23c53efbbd8b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d929141954cb43f68673dd898e72e41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f383b3e6c8a02917f82e23c53efbbd8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 7581, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5c62e0314596b1c49ad67dcd1bf0fda0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f383b3e6c8a02917f82e23c53efbbd8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 7581, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]




if __name__ == '__main__':
    unittest.main()