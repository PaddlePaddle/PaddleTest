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
        return True
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



class PrimitiveOp_7c31399550ef202d07270f8162e49c66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de59dc97cef5f6a56c3c61b7e72315bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c31399550ef202d07270f8162e49c66
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_de59dc97cef5f6a56c3c61b7e72315bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c31399550ef202d07270f8162e49c66
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_31af42573a883d58f3432c9419cf8682(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ad70f532b7e33bfb2273b28e0639bf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31af42573a883d58f3432c9419cf8682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ad70f532b7e33bfb2273b28e0639bf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31af42573a883d58f3432c9419cf8682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_de944bb40bb74a45d9d908ccb174da25(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e46ab39d246c45e4c3aa7d7546234764(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de944bb40bb74a45d9d908ccb174da25
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e46ab39d246c45e4c3aa7d7546234764(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de944bb40bb74a45d9d908ccb174da25
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e6b5805ec7af0d88bfb6e9accc93507e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_08063a1e9d13215f5e743ca0c03f21fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b5805ec7af0d88bfb6e9accc93507e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08063a1e9d13215f5e743ca0c03f21fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b5805ec7af0d88bfb6e9accc93507e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cc87839e84a9ac19e1f82b6ff576d99f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f6df38d2e4f04165c9919aef9fcfd30b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc87839e84a9ac19e1f82b6ff576d99f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f6df38d2e4f04165c9919aef9fcfd30b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc87839e84a9ac19e1f82b6ff576d99f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6aaf2f758df5adbe4e39f7a29b963dcc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_93aa855f52c4fb7e543782b72ba0babb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aaf2f758df5adbe4e39f7a29b963dcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_93aa855f52c4fb7e543782b72ba0babb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aaf2f758df5adbe4e39f7a29b963dcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_de59dc97cef5f6a56c3c61b7e72315bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c31399550ef202d07270f8162e49c66
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_de59dc97cef5f6a56c3c61b7e72315bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c31399550ef202d07270f8162e49c66
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ec36b70774ed7c07b274b3f151805f18(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9dbb1b9bc59316fd1fec1806fa10fa3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec36b70774ed7c07b274b3f151805f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9dbb1b9bc59316fd1fec1806fa10fa3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec36b70774ed7c07b274b3f151805f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_58898b83d91a13f5b58fc4b0af582f8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0397f8714c66438875dfdff445b80e3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58898b83d91a13f5b58fc4b0af582f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0397f8714c66438875dfdff445b80e3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58898b83d91a13f5b58fc4b0af582f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8a639d9f3eaddafd08d066731de3c724(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1787, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1787, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_957de98d62c0bcf546191334d2189284(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a639d9f3eaddafd08d066731de3c724
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_957de98d62c0bcf546191334d2189284(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a639d9f3eaddafd08d066731de3c724
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_957de98d62c0bcf546191334d2189284(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a639d9f3eaddafd08d066731de3c724
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_957de98d62c0bcf546191334d2189284(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a639d9f3eaddafd08d066731de3c724
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b16e12eb901d5e13c56f334912243682(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_498dba954485477ef34f49b4ba03f8f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b16e12eb901d5e13c56f334912243682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_498dba954485477ef34f49b4ba03f8f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b16e12eb901d5e13c56f334912243682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5f60b8e268f203712cdfd228e8f3ad2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2b948f0fe4d33f737b8b0b689618df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f60b8e268f203712cdfd228e8f3ad2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2b948f0fe4d33f737b8b0b689618df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f60b8e268f203712cdfd228e8f3ad2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_483c8d6e38258822c3c09154d493ccd3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d2a18d2e765119a8dea648541271961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_483c8d6e38258822c3c09154d493ccd3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d2a18d2e765119a8dea648541271961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_483c8d6e38258822c3c09154d493ccd3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e6060a67461a022847a401eb2a69a853(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_76d6ac7ce66d2678e511e7344d80f6cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6060a67461a022847a401eb2a69a853
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76d6ac7ce66d2678e511e7344d80f6cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6060a67461a022847a401eb2a69a853
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_770bf8419b3ec186116d5948a10b0232(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fba253f9695d6a7e8a766e71d6c924aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_770bf8419b3ec186116d5948a10b0232
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.340059757232666], [0.387004017829895], [0.4251270294189453], [0.4898195266723633], [0.31481853127479553], [0.4277181327342987], [0.0343017652630806], [0.3381534516811371], [0.09708041697740555]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.14785392582416534], [0.2970240116119385], [0.49885281920433044], [0.40843749046325684], [0.3837003707885742], [0.23476886749267578], [0.007210989482700825], [0.48448464274406433], [0.25622862577438354]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_643389bbdb734c6dab193ff99fd7ae4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_770bf8419b3ec186116d5948a10b0232
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.29957494139671326], [0.36993101239204407], [0.43416088819503784], [0.06923043727874756], [0.01820189692080021], [0.030403364449739456], [0.3196682929992676], [0.24978111684322357], [0.1863652914762497]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.08313236385583878], [0.07048333436250687], [0.31889981031417847], [0.3709265887737274], [0.33858680725097656], [0.4856625199317932], [0.4982873797416687], [0.249019056558609], [0.03643830493092537]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_c7c9c0e6fe7c251b64951915babf3b2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_770bf8419b3ec186116d5948a10b0232
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39222896099090576], [0.4460776746273041], [0.2511337995529175], [0.4043271541595459], [0.4428388476371765], [0.019881412386894226], [0.4657576084136963], [0.4012521505355835], [0.40715867280960083]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3243171274662018], [0.49249011278152466], [0.2755092680454254], [0.2209128737449646], [0.23234640061855316], [0.21800071001052856], [0.1990540474653244], [0.06053031235933304], [0.49496304988861084]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_f0280299283919f793957442e9af5fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_770bf8419b3ec186116d5948a10b0232
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3726502060890198], [0.35267090797424316], [0.026435574516654015], [0.40623462200164795], [0.3046559691429138], [0.4644503593444824], [0.2115304172039032], [0.4490237534046173], [0.2872993052005768]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.45809561014175415], [0.0487419031560421], [0.27907347679138184], [0.3332546353340149], [0.025723274797201157], [0.2000621259212494], [0.2891344726085663], [0.06198172643780708], [0.0746815949678421]], dtype='float32').reshape([9, 1]),
        ]


class PrimitiveOp_4cd688dd306333d5b520353eb3cc09fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5585, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5585, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c99b21e8e996111826e13e0b406f0eaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cd688dd306333d5b520353eb3cc09fa
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c99b21e8e996111826e13e0b406f0eaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cd688dd306333d5b520353eb3cc09fa
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c99b21e8e996111826e13e0b406f0eaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cd688dd306333d5b520353eb3cc09fa
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c99b21e8e996111826e13e0b406f0eaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cd688dd306333d5b520353eb3cc09fa
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_09afba4c000918e41517aed446418de5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            paddle.static.InputSpec(shape=[6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a362341ffa71db421338cd1ec43eea2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.446471244096756, 0.14341187477111816, 0.4889226257801056, 0.025714602321386337, 0.20853398740291595, 0.15274621546268463], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1827273815870285, 0.4455740749835968, 0.4816020429134369, 0.39720895886421204, 0.35945671796798706, 0.4984632730484009], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_5a7dafb19e25f249dea73740576fb4c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2992474138736725, 0.2878594994544983, 0.47377514839172363, 0.05680272728204727, 0.043566759675741196, 0.43330734968185425], dtype='float32').reshape([6]),
            paddle.to_tensor([0.011204774491488934, 0.1125684455037117, 0.10057477653026581, 0.2895292341709137, 0.3072856068611145, 0.31992271542549133], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_5bc3513a143cbd5dd0bfc50f7cb1aeb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.446471244096756, 0.14341187477111816, 0.4889226257801056, 0.025714602321386337, 0.20853398740291595, 0.15274621546268463], dtype='float32').reshape([6]),
            paddle.to_tensor([0.22107045352458954, 0.4040623903274536, 0.46366026997566223, 0.043019529432058334, 0.13830581307411194, 0.4344382584095001], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_cd7a42f582d095cbf3f09ed0e0d22189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2992474138736725, 0.2878594994544983, 0.47377514839172363, 0.05680272728204727, 0.043566759675741196, 0.43330734968185425], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3741452395915985, 0.11825495213270187, 0.027543269097805023, 0.20650611817836761, 0.1986602246761322, 0.02518189512193203], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_2cf48ab6e8cb771bf1b447b79d36ce72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.446471244096756, 0.4455740749835968, 0.4889226257801056, 0.39720895886421204, 0.35945671796798706, 0.4984632730484009], dtype='float32').reshape([6]),
            paddle.to_tensor([0.21949312090873718, 0.0012134052813053131, 0.4766252934932709, 0.11636532098054886, 0.2342921495437622, 0.4063635468482971], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_671d8272f7d74bb84c3475f26163e465(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2992474138736725, 0.2878594994544983, 0.47377514839172363, 0.2895292341709137, 0.3072856068611145, 0.43330734968185425], dtype='float32').reshape([6]),
            paddle.to_tensor([0.031546175479888916, 0.1145879477262497, 0.332019567489624, 0.13446305692195892, 0.41516128182411194, 0.1730593889951706], dtype='float32').reshape([6]),
        ]


class PrimitiveOp_9b59bc0287d6da92ad07b7f9138a40bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1774, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1774, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_61d62fe15e120128368e5ca88ee576b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b59bc0287d6da92ad07b7f9138a40bb
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_61d62fe15e120128368e5ca88ee576b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b59bc0287d6da92ad07b7f9138a40bb
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_61d62fe15e120128368e5ca88ee576b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b59bc0287d6da92ad07b7f9138a40bb
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_61d62fe15e120128368e5ca88ee576b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b59bc0287d6da92ad07b7f9138a40bb
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fd9731f87a86b54aba40f818340e1c5f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 38, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d4ecb1572d2ca345be6ade03a2fccbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd9731f87a86b54aba40f818340e1c5f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.000000013351432e-10], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_08063a1e9d13215f5e743ca0c03f21fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b5805ec7af0d88bfb6e9accc93507e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08063a1e9d13215f5e743ca0c03f21fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b5805ec7af0d88bfb6e9accc93507e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2b948f0fe4d33f737b8b0b689618df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f60b8e268f203712cdfd228e8f3ad2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2b948f0fe4d33f737b8b0b689618df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f60b8e268f203712cdfd228e8f3ad2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c10b83757db0f3b02d6fa18bb66bf765(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1501, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1501, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3529d92cb927980537a1ef38e30f8a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c10b83757db0f3b02d6fa18bb66bf765
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3529d92cb927980537a1ef38e30f8a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c10b83757db0f3b02d6fa18bb66bf765
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3529d92cb927980537a1ef38e30f8a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c10b83757db0f3b02d6fa18bb66bf765
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3529d92cb927980537a1ef38e30f8a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c10b83757db0f3b02d6fa18bb66bf765
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_93aa855f52c4fb7e543782b72ba0babb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aaf2f758df5adbe4e39f7a29b963dcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_93aa855f52c4fb7e543782b72ba0babb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aaf2f758df5adbe4e39f7a29b963dcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ad70f532b7e33bfb2273b28e0639bf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31af42573a883d58f3432c9419cf8682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ad70f532b7e33bfb2273b28e0639bf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31af42573a883d58f3432c9419cf8682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bda4b337647876c6122658aac83ee09d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fbe7d5a338837a45d1874cc9df009a6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bda4b337647876c6122658aac83ee09d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.28265708684921265]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3178286850452423]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_91897567a9f4cfae39e9dec180c27421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bda4b337647876c6122658aac83ee09d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06415817886590958]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.43161070346832275]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_eea7a9095fb43cfe9ca30ed9df6c9592(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bda4b337647876c6122658aac83ee09d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.021477889269590378]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.4371948540210724]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_5f0814ad7cbeae2cd847c2af5a9ea492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bda4b337647876c6122658aac83ee09d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.36230334639549255]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3298352062702179]], dtype='float32').reshape([1, 1]),
        ]


class PrimitiveOp_29df4571177cd912e25c2718ed354418(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_90d5eb9e88a177c03ce4a1753301e6cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29df4571177cd912e25c2718ed354418
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.35102102160453796], [0.4810973107814789], [0.48731809854507446], [0.059361398220062256], [0.40835464000701904], [0.47783151268959045]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.4764886200428009], [0.21417561173439026], [0.3296174705028534], [0.20144562423229218], [0.09571299701929092], [0.09185384213924408]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_acc6fe95e3f283aa5e8b3cb818fa4ed1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29df4571177cd912e25c2718ed354418
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10640901327133179], [0.15210367739200592], [0.058579739183187485], [0.04452253505587578], [0.053740326315164566], [0.4524851441383362]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.29171887040138245], [0.3921818137168884], [0.2787969708442688], [0.39924129843711853], [0.09039394557476044], [0.48042625188827515]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_54041aadff1eacfc8d6f3bee37796b3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29df4571177cd912e25c2718ed354418
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33286547660827637], [0.028109095990657806], [0.37021663784980774], [0.08470499515533447], [0.0046716793440282345], [0.19173109531402588]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3928629457950592], [0.3117378354072571], [0.14381171762943268], [0.40953099727630615], [0.0882832407951355], [0.15646472573280334]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_ee76b97ae9714579e12c4ae12bd1fc42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29df4571177cd912e25c2718ed354418
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.26473021507263184], [0.4301255941390991], [0.009996733628213406], [0.43156635761260986], [0.38260313868522644], [0.05900372192263603]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.43764728307724], [0.4305558502674103], [0.3785059452056885], [0.4988960921764374], [0.02951250597834587], [0.22250328958034515]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_498dba954485477ef34f49b4ba03f8f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b16e12eb901d5e13c56f334912243682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_498dba954485477ef34f49b4ba03f8f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b16e12eb901d5e13c56f334912243682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e46ab39d246c45e4c3aa7d7546234764(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de944bb40bb74a45d9d908ccb174da25
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e46ab39d246c45e4c3aa7d7546234764(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de944bb40bb74a45d9d908ccb174da25
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d60413012f86daa89a975396d30c2d45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2049, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2049, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e42f52c914cff1ee50f0271bd397512d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d60413012f86daa89a975396d30c2d45
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e42f52c914cff1ee50f0271bd397512d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d60413012f86daa89a975396d30c2d45
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e42f52c914cff1ee50f0271bd397512d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d60413012f86daa89a975396d30c2d45
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e42f52c914cff1ee50f0271bd397512d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d60413012f86daa89a975396d30c2d45
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e46c4d974bba9fa83ba73424a9c938bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8fdb8fcbe31c79561c2f38d1b7cb468b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e46c4d974bba9fa83ba73424a9c938bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fdb8fcbe31c79561c2f38d1b7cb468b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e46c4d974bba9fa83ba73424a9c938bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3631381f254aafe756ca93a6ea589bcc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4634, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4634, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6d02a0e7d302a609a965d3ca838c758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3631381f254aafe756ca93a6ea589bcc
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6d02a0e7d302a609a965d3ca838c758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3631381f254aafe756ca93a6ea589bcc
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6d02a0e7d302a609a965d3ca838c758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3631381f254aafe756ca93a6ea589bcc
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6d02a0e7d302a609a965d3ca838c758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3631381f254aafe756ca93a6ea589bcc
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6788ae850d9c6a6a03669923efaf30bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1000, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1000, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40e3fed0002cea26c0ed0b26bdb378d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6788ae850d9c6a6a03669923efaf30bc
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_40e3fed0002cea26c0ed0b26bdb378d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6788ae850d9c6a6a03669923efaf30bc
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_40e3fed0002cea26c0ed0b26bdb378d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6788ae850d9c6a6a03669923efaf30bc
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_40e3fed0002cea26c0ed0b26bdb378d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6788ae850d9c6a6a03669923efaf30bc
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f6df38d2e4f04165c9919aef9fcfd30b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc87839e84a9ac19e1f82b6ff576d99f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f6df38d2e4f04165c9919aef9fcfd30b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc87839e84a9ac19e1f82b6ff576d99f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d2a18d2e765119a8dea648541271961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_483c8d6e38258822c3c09154d493ccd3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d2a18d2e765119a8dea648541271961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_483c8d6e38258822c3c09154d493ccd3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_23bd35d07a8597402908f29ca134acd7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bea13749170a78e9e08a8d9ce86743ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23bd35d07a8597402908f29ca134acd7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10822069644927979], [0.06122198328375816], [0.35097381472587585], [0.4020307958126068], [0.36489900946617126]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.24273258447647095], [0.08545062690973282], [0.11754946410655975], [0.3820875585079193], [0.32701370120048523]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_85b7c76704bddb4d4c0cea62a67f5598(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23bd35d07a8597402908f29ca134acd7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1466745138168335], [0.2202141284942627], [0.20410498976707458], [0.19355495274066925], [0.125055193901062]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.15814867615699768], [0.42982131242752075], [0.37481895089149475], [0.38867270946502686], [0.0035878224298357964]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_6f16a4e3627a6f7eb1c40f550d164a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23bd35d07a8597402908f29ca134acd7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3062000572681427], [0.3130226135253906], [0.17877483367919922], [0.19642746448516846], [0.023015061393380165]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.35707125067710876], [0.03344854712486267], [0.07517310976982117], [0.34104833006858826], [0.4732273817062378]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_55115804c8468d4285d48c7006a3da42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23bd35d07a8597402908f29ca134acd7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.02279740385711193], [0.34154266119003296], [0.30135348439216614], [0.4762178957462311], [0.05762705206871033]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.20563507080078125], [0.15860792994499207], [0.21742399036884308], [0.2578918933868408], [0.23720066249370575]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_0397f8714c66438875dfdff445b80e3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58898b83d91a13f5b58fc4b0af582f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0397f8714c66438875dfdff445b80e3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58898b83d91a13f5b58fc4b0af582f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76d6ac7ce66d2678e511e7344d80f6cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6060a67461a022847a401eb2a69a853
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76d6ac7ce66d2678e511e7344d80f6cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6060a67461a022847a401eb2a69a853
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_76211934d2a84fd2ee5cf8ea4fc0ccfe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_781c06be7208b8e8ad96cd34e511f934(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76211934d2a84fd2ee5cf8ea4fc0ccfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_781c06be7208b8e8ad96cd34e511f934(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76211934d2a84fd2ee5cf8ea4fc0ccfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2af07131430f7a7d129b2f4d21ccd517(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2382, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2382, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_78d93b36eb6fa288814a74d2069d1d9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af07131430f7a7d129b2f4d21ccd517
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78d93b36eb6fa288814a74d2069d1d9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af07131430f7a7d129b2f4d21ccd517
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78d93b36eb6fa288814a74d2069d1d9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af07131430f7a7d129b2f4d21ccd517
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78d93b36eb6fa288814a74d2069d1d9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af07131430f7a7d129b2f4d21ccd517
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3bdd66c14e6047e8dedec42c169be58b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2976, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2976, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c5fa3b93c86c51736c2b7876fef22491(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bdd66c14e6047e8dedec42c169be58b
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5fa3b93c86c51736c2b7876fef22491(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bdd66c14e6047e8dedec42c169be58b
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5fa3b93c86c51736c2b7876fef22491(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bdd66c14e6047e8dedec42c169be58b
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5fa3b93c86c51736c2b7876fef22491(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bdd66c14e6047e8dedec42c169be58b
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_86a8bc8eb9b9cc8b63c97c19a4a3e205(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3753, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3753, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1bc5f492352cdfea6d7e98db27bfae16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86a8bc8eb9b9cc8b63c97c19a4a3e205
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1bc5f492352cdfea6d7e98db27bfae16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86a8bc8eb9b9cc8b63c97c19a4a3e205
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1bc5f492352cdfea6d7e98db27bfae16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86a8bc8eb9b9cc8b63c97c19a4a3e205
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1bc5f492352cdfea6d7e98db27bfae16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86a8bc8eb9b9cc8b63c97c19a4a3e205
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fdb8fcbe31c79561c2f38d1b7cb468b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e46c4d974bba9fa83ba73424a9c938bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fdb8fcbe31c79561c2f38d1b7cb468b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e46c4d974bba9fa83ba73424a9c938bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f5cc331f0a0f34974d8726d249ced0cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d72a81991ffae0e3ba37e8fd5afdb095(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5cc331f0a0f34974d8726d249ced0cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17840133607387543], [0.15334481000900269], [0.35247161984443665], [0.10544773191213608]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3934442102909088], [0.42687827348709106], [0.24674589931964874], [0.20445282757282257]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_4b13be5b7f26295f88c3f1b5db89e386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5cc331f0a0f34974d8726d249ced0cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.42456135153770447], [0.41002196073532104], [0.4354965388774872], [0.12312145531177521]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4565404951572418], [0.4529203772544861], [0.1775546818971634], [0.31460875272750854]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_7cf550fc22f1c6af5348439b75744e1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5cc331f0a0f34974d8726d249ced0cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4963822662830353], [0.2475523203611374], [0.11298491805791855], [0.2765282988548279]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.25048601627349854], [0.43678945302963257], [0.09424278885126114], [0.2916932702064514]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_94099fb4ea860dbb40c7663495703dfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5cc331f0a0f34974d8726d249ced0cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2910788357257843], [0.030457817018032074], [0.4478048086166382], [0.3236304521560669]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.1932741403579712], [0.1818334460258484], [0.12248531728982925], [0.3949283957481384]], dtype='float32').reshape([4, 1]),
        ]


class PrimitiveOp_d4553a6766d7dd6b85b77d4fac5fef81(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1995, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1995, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_853aefaf376d7a149e8782aa5df6e8f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4553a6766d7dd6b85b77d4fac5fef81
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_853aefaf376d7a149e8782aa5df6e8f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4553a6766d7dd6b85b77d4fac5fef81
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_853aefaf376d7a149e8782aa5df6e8f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4553a6766d7dd6b85b77d4fac5fef81
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_853aefaf376d7a149e8782aa5df6e8f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4553a6766d7dd6b85b77d4fac5fef81
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_781c06be7208b8e8ad96cd34e511f934(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76211934d2a84fd2ee5cf8ea4fc0ccfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_781c06be7208b8e8ad96cd34e511f934(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76211934d2a84fd2ee5cf8ea4fc0ccfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9dbb1b9bc59316fd1fec1806fa10fa3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec36b70774ed7c07b274b3f151805f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9dbb1b9bc59316fd1fec1806fa10fa3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec36b70774ed7c07b274b3f151805f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f7f86dd05442e6612faa46448001ab97(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc4a970b7a1a4ae8b273d275e52edf0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7f86dd05442e6612faa46448001ab97
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc4a970b7a1a4ae8b273d275e52edf0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7f86dd05442e6612faa46448001ab97
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ba396860e456626c43664dedebf8f049(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4185, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_944b696e5f4582fe93666d5f25defa99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba396860e456626c43664dedebf8f049
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_944b696e5f4582fe93666d5f25defa99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba396860e456626c43664dedebf8f049
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_944b696e5f4582fe93666d5f25defa99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba396860e456626c43664dedebf8f049
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_944b696e5f4582fe93666d5f25defa99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba396860e456626c43664dedebf8f049
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc4a970b7a1a4ae8b273d275e52edf0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7f86dd05442e6612faa46448001ab97
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc4a970b7a1a4ae8b273d275e52edf0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7f86dd05442e6612faa46448001ab97
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()