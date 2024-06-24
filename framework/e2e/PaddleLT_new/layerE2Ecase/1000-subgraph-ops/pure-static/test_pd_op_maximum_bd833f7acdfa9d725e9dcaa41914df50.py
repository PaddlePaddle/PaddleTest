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


class PrimitiveOp_2e749af6ce4030ecf22b9e2828df731e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1786, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1786, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e93563ef8ba71969cde0137d48488637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e749af6ce4030ecf22b9e2828df731e
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e93563ef8ba71969cde0137d48488637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e749af6ce4030ecf22b9e2828df731e
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e93563ef8ba71969cde0137d48488637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e749af6ce4030ecf22b9e2828df731e
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e93563ef8ba71969cde0137d48488637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e749af6ce4030ecf22b9e2828df731e
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_76f5b6446d47f226081ce7da09327636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_770bf8419b3ec186116d5948a10b0232
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2622634470462799], [0.2463095784187317], [0.40173643827438354], [0.3592231869697571], [0.3121943771839142], [0.14940416812896729], [0.09890615940093994], [0.3055746257305145], [0.06899479031562805]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.08312228322029114], [0.009725884534418583], [0.31125256419181824], [0.18358245491981506], [0.14136718213558197], [0.22684775292873383], [0.2601875066757202], [0.1880447119474411], [0.2560676038265228]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_291126df69a9407ba6cfd0f20c262d3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_770bf8419b3ec186116d5948a10b0232
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3488651216030121], [0.4772522747516632], [0.07919348776340485], [0.36751043796539307], [0.10987333208322525], [0.11728937923908234], [0.13010436296463013], [0.21030104160308838], [0.2969972491264343]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.07335024327039719], [0.072712242603302], [0.08202075958251953], [0.35215020179748535], [0.43682003021240234], [0.356018990278244], [0.25737234950065613], [0.4559156596660614], [0.34402239322662354]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_75103da3498c211285954a9f907302aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_770bf8419b3ec186116d5948a10b0232
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0944250226020813], [0.43199148774147034], [0.30663737654685974], [0.30789077281951904], [0.017560848966240883], [0.4452926218509674], [0.3800290524959564], [0.212552011013031], [0.06295822560787201]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.4870230257511139], [0.41575801372528076], [0.09537261724472046], [0.13353395462036133], [0.23935119807720184], [0.10870978981256485], [0.15444859862327576], [0.14588458836078644], [0.23597005009651184]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_99738194d08e399372b1c0b984c7dfaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_770bf8419b3ec186116d5948a10b0232
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.46759042143821716], [0.05409188196063042], [0.31611233949661255], [0.4931334853172302], [0.3475216031074524], [0.33039727807044983], [0.12327843904495239], [0.307874858379364], [0.3336060643196106]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.15379993617534637], [0.34884482622146606], [0.3303232789039612], [0.3350580632686615], [0.24807313084602356], [0.003155889455229044], [0.48330458998680115], [0.15179887413978577], [0.044809091836214066]], dtype='float32').reshape([9, 1]),
        ]


class PrimitiveOp_194e85690ec602ef9a49c1e3f19b13db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5529, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5529, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8bf4b28b87861f3e2fe5ca0e1a04a630(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_194e85690ec602ef9a49c1e3f19b13db
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bf4b28b87861f3e2fe5ca0e1a04a630(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_194e85690ec602ef9a49c1e3f19b13db
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bf4b28b87861f3e2fe5ca0e1a04a630(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_194e85690ec602ef9a49c1e3f19b13db
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bf4b28b87861f3e2fe5ca0e1a04a630(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_194e85690ec602ef9a49c1e3f19b13db
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_968ab936839dcdbdb62ff2c2d4cbc52b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09849236905574799, 0.377002090215683, 0.03728121519088745, 0.03544168174266815, 0.3461141586303711, 0.37609827518463135], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4773276150226593, 0.4900837242603302, 0.31976962089538574, 0.23725539445877075, 0.3398610055446625, 0.47337961196899414], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_5e09cd624eb6418a4e2cd21dfe3ebba7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10391133278608322, 0.45978280901908875, 0.49666598439216614, 0.21115154027938843, 0.31940513849258423, 0.33928123116493225], dtype='float32').reshape([6]),
            paddle.to_tensor([0.12860596179962158, 0.07729382812976837, 0.4407457411289215, 0.08533225953578949, 0.47597774863243103, 0.44470906257629395], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_549fe087a2c70c2ab2d162fde0dcc320(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09849236905574799, 0.377002090215683, 0.03728121519088745, 0.03544168174266815, 0.3461141586303711, 0.37609827518463135], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2667119801044464, 0.46009713411331177, 0.48630520701408386, 0.3197534382343292, 0.08830360323190689, 0.18910035490989685], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_6dec17606b96cb8ba94512305fe5e4fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10391133278608322, 0.45978280901908875, 0.49666598439216614, 0.21115154027938843, 0.31940513849258423, 0.33928123116493225], dtype='float32').reshape([6]),
            paddle.to_tensor([0.33211463689804077, 0.06320647895336151, 0.03932555019855499, 0.3673308491706848, 0.05595792829990387, 0.18198029696941376], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_abc7a7ee62de699b26d6da5166a81fcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4773276150226593, 0.4900837242603302, 0.31976962089538574, 0.23725539445877075, 0.3461141586303711, 0.47337961196899414], dtype='float32').reshape([6]),
            paddle.to_tensor([0.09207890182733536, 0.23877081274986267, 0.14726030826568604, 0.3021329939365387, 0.4564148485660553, 0.3768220543861389], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ecb855f705a684a8a0da4d44cf1c497a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12860596179962158, 0.45978280901908875, 0.49666598439216614, 0.21115154027938843, 0.47597774863243103, 0.44470906257629395], dtype='float32').reshape([6]),
            paddle.to_tensor([0.32528331875801086, 0.010651743039488792, 0.30837002396583557, 0.4661112129688263, 0.16577965021133423, 0.012630387209355831], dtype='float32').reshape([6]),
        ]


class PrimitiveOp_eae9cf58de2cd730d9a2fa36a1f89fba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1767, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1767, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69a42ffbe9f83261491e5fe32e30fbc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eae9cf58de2cd730d9a2fa36a1f89fba
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69a42ffbe9f83261491e5fe32e30fbc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eae9cf58de2cd730d9a2fa36a1f89fba
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69a42ffbe9f83261491e5fe32e30fbc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eae9cf58de2cd730d9a2fa36a1f89fba
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69a42ffbe9f83261491e5fe32e30fbc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eae9cf58de2cd730d9a2fa36a1f89fba
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_276fabaee3a7e1871b1fa69358f66b9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1490, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1490, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f168e2a98ea62bf7b79e199fdb8a177c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_276fabaee3a7e1871b1fa69358f66b9b
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f168e2a98ea62bf7b79e199fdb8a177c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_276fabaee3a7e1871b1fa69358f66b9b
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f168e2a98ea62bf7b79e199fdb8a177c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_276fabaee3a7e1871b1fa69358f66b9b
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f168e2a98ea62bf7b79e199fdb8a177c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_276fabaee3a7e1871b1fa69358f66b9b
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f77a0fd48908d29051662e8be0cf6483(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bda4b337647876c6122658aac83ee09d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10528379678726196]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3151826858520508]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_5bd5d8cfc3eb57698d4209fe77a8518c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bda4b337647876c6122658aac83ee09d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07341162115335464]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.21531380712985992]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_6cbecbc3127fca2a9839fae8586cbb48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bda4b337647876c6122658aac83ee09d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.41494059562683105]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3083866834640503]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_fa2fc54a896a786f4524f8914e566025(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bda4b337647876c6122658aac83ee09d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16263581812381744]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3545854091644287]], dtype='float32').reshape([1, 1]),
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


class TestPrimitiveOp_906eb4ac93bfff700ad10fa5dfcfed73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29df4571177cd912e25c2718ed354418
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10129960626363754], [0.42914801836013794], [0.3438938558101654], [0.010731011629104614], [0.2497348189353943], [0.2733650505542755]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.09891922771930695], [0.4956405460834503], [0.13846367597579956], [0.49526292085647583], [0.09620732814073563], [0.3659272789955139]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_3f267d1d3132b0e9514f6af79c6feeee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29df4571177cd912e25c2718ed354418
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.31633827090263367], [0.34771016240119934], [0.2228093296289444], [0.22192063927650452], [0.43225693702697754], [0.3838341534137726]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3794229328632355], [0.35555100440979004], [0.0830652266740799], [0.07343913614749908], [0.06483077257871628], [0.41214850544929504]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_e83da45a6dec268e81f8526540dbf6c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29df4571177cd912e25c2718ed354418
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.376817524433136], [0.45837101340293884], [0.4508039653301239], [0.24826665222644806], [0.04853551834821701], [0.25887274742126465]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.04222099483013153], [0.4947202205657959], [0.46678364276885986], [0.2570192813873291], [0.16303293406963348], [0.3793601393699646]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_1d2991f2cce3604a2410e4578bdbd40a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29df4571177cd912e25c2718ed354418
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2849709689617157], [0.4665500521659851], [0.4074956178665161], [0.22639164328575134], [0.21449042856693268], [0.04977961629629135]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3085649907588959], [0.11111163347959518], [0.23699478805065155], [0.22173336148262024], [0.05751029774546623], [0.39112553000450134]], dtype='float32').reshape([6, 1]),
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


class PrimitiveOp_59b250d8299af8d4de3b1aab8060453c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2010, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2010, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_134b0a8139c842db1be9dd7a1844977d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59b250d8299af8d4de3b1aab8060453c
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_134b0a8139c842db1be9dd7a1844977d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59b250d8299af8d4de3b1aab8060453c
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_134b0a8139c842db1be9dd7a1844977d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59b250d8299af8d4de3b1aab8060453c
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_134b0a8139c842db1be9dd7a1844977d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59b250d8299af8d4de3b1aab8060453c
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_ae66f3592ba40bc6135db1d85808ef19(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4663, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4663, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cae215876f22013b268539aa3aa3138e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae66f3592ba40bc6135db1d85808ef19
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cae215876f22013b268539aa3aa3138e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae66f3592ba40bc6135db1d85808ef19
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cae215876f22013b268539aa3aa3138e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae66f3592ba40bc6135db1d85808ef19
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cae215876f22013b268539aa3aa3138e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae66f3592ba40bc6135db1d85808ef19
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bf9ca342bf0b43208608c9d6ec9ac9f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1090, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1090, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9cdf821bc9832c995a774d8df07911f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf9ca342bf0b43208608c9d6ec9ac9f9
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9cdf821bc9832c995a774d8df07911f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf9ca342bf0b43208608c9d6ec9ac9f9
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9cdf821bc9832c995a774d8df07911f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf9ca342bf0b43208608c9d6ec9ac9f9
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9cdf821bc9832c995a774d8df07911f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf9ca342bf0b43208608c9d6ec9ac9f9
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_3e8289361c271546fa355a106aeb7d5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23bd35d07a8597402908f29ca134acd7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06819140911102295], [0.0015212241560220718], [0.29410940408706665], [0.0312783420085907], [0.05078583583235741]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.35739219188690186], [0.11215049773454666], [0.24850419163703918], [0.08986014872789383], [0.42450934648513794]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_91288ee755d9d2e695b6a3c6c4f3eaea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23bd35d07a8597402908f29ca134acd7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.34516990184783936], [0.14900963008403778], [0.23224982619285583], [0.4999895989894867], [0.029579993337392807]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.2559264004230499], [0.4242517352104187], [0.3585534393787384], [0.4039344787597656], [0.33067619800567627]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_8666a57d632049c4b7378af02ba363ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23bd35d07a8597402908f29ca134acd7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0807226225733757], [0.26894018054008484], [0.30494368076324463], [0.27619537711143494], [0.428976833820343]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.42150676250457764], [0.40049660205841064], [0.22837966680526733], [0.09048057347536087], [0.3214188814163208]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_0ec880ff9c64e359d3e4f78465593f26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23bd35d07a8597402908f29ca134acd7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.23073291778564453], [0.08885031193494797], [0.22880247235298157], [0.4462369680404663], [0.41620945930480957]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.15623098611831665], [0.23012053966522217], [0.0010281642898917198], [0.315184623003006], [0.44070783257484436]], dtype='float32').reshape([5, 1]),
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


class PrimitiveOp_acad4c209bb14af9f39f7b757ce42730(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2374, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2374, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cdbbcb9762d7fa4ccdf0461629e65e03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acad4c209bb14af9f39f7b757ce42730
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cdbbcb9762d7fa4ccdf0461629e65e03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acad4c209bb14af9f39f7b757ce42730
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cdbbcb9762d7fa4ccdf0461629e65e03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acad4c209bb14af9f39f7b757ce42730
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cdbbcb9762d7fa4ccdf0461629e65e03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acad4c209bb14af9f39f7b757ce42730
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_838a2b3bbd7bf419818119874be18ca5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3058, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3058, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_35e53d783e0ac6b8d250b8c6d023f8fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_838a2b3bbd7bf419818119874be18ca5
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35e53d783e0ac6b8d250b8c6d023f8fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_838a2b3bbd7bf419818119874be18ca5
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35e53d783e0ac6b8d250b8c6d023f8fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_838a2b3bbd7bf419818119874be18ca5
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35e53d783e0ac6b8d250b8c6d023f8fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_838a2b3bbd7bf419818119874be18ca5
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7bf5c22396cb63507e5e9147896d3ea9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3793, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3793, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e035682a0e30ed0a9e0edc5a6a96c345(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bf5c22396cb63507e5e9147896d3ea9
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e035682a0e30ed0a9e0edc5a6a96c345(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bf5c22396cb63507e5e9147896d3ea9
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e035682a0e30ed0a9e0edc5a6a96c345(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bf5c22396cb63507e5e9147896d3ea9
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e035682a0e30ed0a9e0edc5a6a96c345(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bf5c22396cb63507e5e9147896d3ea9
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_4e1d16849c4a4c1aad3be289ea7d4dfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5cc331f0a0f34974d8726d249ced0cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06521501392126083], [0.034968309104442596], [0.26277071237564087], [0.01609121635556221]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.339884489774704], [0.06699070334434509], [0.30670201778411865], [0.038516294211149216]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_2e7b6db620a5abd5a34285f554877d04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5cc331f0a0f34974d8726d249ced0cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17842870950698853], [0.3201243281364441], [0.30915895104408264], [0.2734312415122986]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3211865723133087], [0.02025376260280609], [0.31398555636405945], [0.023009276017546654]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_c8e6f9779fcc0884dff5bc3e0541909d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5cc331f0a0f34974d8726d249ced0cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11390042304992676], [0.4904465973377228], [0.13921834528446198], [0.496462345123291]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.19393745064735413], [0.1851010024547577], [0.08248813450336456], [0.12678833305835724]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_0729b0219b3cb07701728cf552297485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5cc331f0a0f34974d8726d249ced0cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3694194257259369], [0.4472231864929199], [0.41353461146354675], [0.07219047099351883]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.020291708409786224], [0.017377078533172607], [0.2168794572353363], [0.03630860522389412]], dtype='float32').reshape([4, 1]),
        ]


class PrimitiveOp_c8061369b71e3a4c57343030b6d9b0f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2042, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2042, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_342498f36bf2b93fe57ad7c9fa968c09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8061369b71e3a4c57343030b6d9b0f5
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_342498f36bf2b93fe57ad7c9fa968c09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8061369b71e3a4c57343030b6d9b0f5
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_342498f36bf2b93fe57ad7c9fa968c09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8061369b71e3a4c57343030b6d9b0f5
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_342498f36bf2b93fe57ad7c9fa968c09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8061369b71e3a4c57343030b6d9b0f5
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
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