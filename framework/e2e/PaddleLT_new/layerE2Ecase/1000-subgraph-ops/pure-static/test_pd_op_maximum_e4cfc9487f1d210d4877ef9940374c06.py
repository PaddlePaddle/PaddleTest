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


class PrimitiveOp_ddf84fa59d49c81912d31feb406bce6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1758, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1758, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1dbe4d43e2f912a1dee058e59ed17b45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddf84fa59d49c81912d31feb406bce6f
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1dbe4d43e2f912a1dee058e59ed17b45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddf84fa59d49c81912d31feb406bce6f
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1dbe4d43e2f912a1dee058e59ed17b45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddf84fa59d49c81912d31feb406bce6f
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1dbe4d43e2f912a1dee058e59ed17b45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddf84fa59d49c81912d31feb406bce6f
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_5bc73d1f40fdd2440f9619ac3b15e896(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_770bf8419b3ec186116d5948a10b0232
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4423205256462097], [0.18592821061611176], [0.2645948827266693], [0.24683165550231934], [0.23438315093517303], [0.4034620225429535], [0.20104055106639862], [0.18984976410865784], [0.18868638575077057]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.12630529701709747], [0.4453306794166565], [0.027548898011446], [0.3977745771408081], [0.25538399815559387], [0.17480947077274323], [0.2219296097755432], [0.3580722510814667], [0.44802966713905334]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_8eb649fdb43030c19d8278f791e1116c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_770bf8419b3ec186116d5948a10b0232
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.24950331449508667], [0.34855425357818604], [0.26499083638191223], [0.0974547415971756], [0.49846959114074707], [0.30478185415267944], [0.2055416703224182], [0.3756718039512634], [0.030764833092689514]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.41259145736694336], [0.3572691082954407], [0.1078881248831749], [0.3949889540672302], [0.296988844871521], [0.4767949879169464], [0.15632693469524384], [0.31878870725631714], [0.17685101926326752]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_6cec66a814dabce8617bbc404c46b71c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_770bf8419b3ec186116d5948a10b0232
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11550065129995346], [0.374041348695755], [0.15645268559455872], [0.2919510304927826], [0.16898958384990692], [0.05408820882439613], [0.11748014390468597], [0.40530943870544434], [0.1729610115289688]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3067001700401306], [0.2585631310939789], [0.4714326858520508], [0.13411420583724976], [0.47448375821113586], [0.4214091897010803], [0.38982850313186646], [0.18900315463542938], [0.18645043671131134]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_01e7dd6c96452458bcfe8a5e4e466929(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_770bf8419b3ec186116d5948a10b0232
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.044602248817682266], [0.11456053704023361], [0.33949682116508484], [0.03393561393022537], [0.4222550392150879], [0.029572542756795883], [0.15619488060474396], [0.0473758690059185], [0.3221006393432617]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.14979442954063416], [0.44975969195365906], [0.3147315979003906], [0.0915440171957016], [0.21444787085056305], [0.03469356149435043], [0.23302797973155975], [0.32175347208976746], [0.4466075897216797]], dtype='float32').reshape([9, 1]),
        ]


class PrimitiveOp_e5a1a0ceb2d5487f2ac8829f1957b07c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5593, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5593, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5a81194e4c2dfeee485bff75a62af0be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5a1a0ceb2d5487f2ac8829f1957b07c
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a81194e4c2dfeee485bff75a62af0be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5a1a0ceb2d5487f2ac8829f1957b07c
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a81194e4c2dfeee485bff75a62af0be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5a1a0ceb2d5487f2ac8829f1957b07c
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a81194e4c2dfeee485bff75a62af0be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5a1a0ceb2d5487f2ac8829f1957b07c
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_e675a4d32fcf674816f3d940371c480c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2925397753715515, 0.4553702473640442, 0.23048850893974304, 0.27912789583206177, 0.4798333942890167, 0.1506512463092804], dtype='float32').reshape([6]),
            paddle.to_tensor([0.44525033235549927, 0.05215751752257347, 0.3241686224937439, 0.16674992442131042, 0.17867626249790192, 0.076868936419487], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_e960f2cdc4ec43b8c4ae6e74df7898d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.40655165910720825, 0.18528297543525696, 0.49778977036476135, 0.06130501255393028, 0.47947970032691956, 0.3735698163509369], dtype='float32').reshape([6]),
            paddle.to_tensor([0.35288164019584656, 0.2496812641620636, 0.19527703523635864, 0.02379102259874344, 0.29160597920417786, 0.1866409331560135], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_799aa19c1585ff4d83e53edb9876c6a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2925397753715515, 0.4553702473640442, 0.23048850893974304, 0.27912789583206177, 0.4798333942890167, 0.1506512463092804], dtype='float32').reshape([6]),
            paddle.to_tensor([0.28722062706947327, 0.21731960773468018, 0.3951013386249542, 0.3923845887184143, 0.3277442157268524, 0.35763120651245117], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_7b9acf05c0821d0bbfa9335eca098807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.40655165910720825, 0.18528297543525696, 0.49778977036476135, 0.06130501255393028, 0.47947970032691956, 0.3735698163509369], dtype='float32').reshape([6]),
            paddle.to_tensor([0.43829435110092163, 0.012103257700800896, 0.42449864745140076, 0.4616372585296631, 0.0976329892873764, 0.0438818633556366], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_82c1632e98e867ced225cc354eaf646c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44525033235549927, 0.4553702473640442, 0.3241686224937439, 0.27912789583206177, 0.4798333942890167, 0.1506512463092804], dtype='float32').reshape([6]),
            paddle.to_tensor([0.18762415647506714, 0.2747806906700134, 0.4200975298881531, 0.3786199986934662, 0.03297353908419609, 0.23518399894237518], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_3c27f2004b324683e5c4cc8d97729ba8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.40655165910720825, 0.2496812641620636, 0.49778977036476135, 0.06130501255393028, 0.47947970032691956, 0.3735698163509369], dtype='float32').reshape([6]),
            paddle.to_tensor([0.18261484801769257, 0.3417728841304779, 0.2622556686401367, 0.1527484655380249, 0.36947619915008545, 0.32382044196128845], dtype='float32').reshape([6]),
        ]


class PrimitiveOp_b3387ec45e26ee6618eb26528572f36b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1763, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1763, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_05b684b98200a04f00680901c4f32a60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3387ec45e26ee6618eb26528572f36b
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05b684b98200a04f00680901c4f32a60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3387ec45e26ee6618eb26528572f36b
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05b684b98200a04f00680901c4f32a60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3387ec45e26ee6618eb26528572f36b
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05b684b98200a04f00680901c4f32a60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3387ec45e26ee6618eb26528572f36b
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_7d454cfe09e7ca5cbf251ff4b967a991(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bda4b337647876c6122658aac83ee09d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.491269052028656]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.4348372220993042]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_e6a0f13206e7c7afef7266c7790675d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bda4b337647876c6122658aac83ee09d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09417319297790527]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.09474091231822968]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_e7d765fffcdf07a45b6de89d7869bcac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bda4b337647876c6122658aac83ee09d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.36167198419570923]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3290674686431885]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_154a471bb7ca4e5614369ec27bf8846f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bda4b337647876c6122658aac83ee09d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1334301382303238]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0780901163816452]], dtype='float32').reshape([1, 1]),
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


class TestPrimitiveOp_a61cfbd5ab234858d15c7abc2078c904(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29df4571177cd912e25c2718ed354418
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.49020639061927795], [0.06694857776165009], [0.37551629543304443], [0.16441957652568817], [0.056271616369485855], [0.4976474642753601]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.03940660506486893], [0.4823766052722931], [0.3955044746398926], [0.22639241814613342], [0.3185456097126007], [0.04834653064608574]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_bc6ebd9105a242e7ed32cfdd6958c09c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29df4571177cd912e25c2718ed354418
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.38628971576690674], [0.3818877637386322], [0.4154316186904907], [0.09654207527637482], [0.48540663719177246], [0.23541131615638733]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.13275480270385742], [0.24877241253852844], [0.14286577701568604], [0.02221701852977276], [0.3493429720401764], [0.074555404484272]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_ea0034f4cab5700d76a7e603b5e4f010(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29df4571177cd912e25c2718ed354418
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.40875673294067383], [0.17066887021064758], [0.32239457964897156], [0.12811346352100372], [0.056965190917253494], [0.37778374552726746]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3079056143760681], [0.3012850880622864], [0.45780348777770996], [0.38056784868240356], [0.45184314250946045], [0.28739556670188904]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_b3017bb4c22ab8ce3e6c358ae05897c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29df4571177cd912e25c2718ed354418
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14837408065795898], [0.1733187437057495], [0.2672334313392639], [0.09106913208961487], [0.15825320780277252], [0.12337625026702881]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.23708929121494293], [0.3412092924118042], [0.4712662696838379], [0.33640754222869873], [0.018445204943418503], [0.33481597900390625]], dtype='float32').reshape([6, 1]),
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


class PrimitiveOp_1d9fa403e72c01a0082c5d062e113482(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2076, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2076, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8f8db8992e2e6003922c43dcc7e6b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d9fa403e72c01a0082c5d062e113482
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e8f8db8992e2e6003922c43dcc7e6b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d9fa403e72c01a0082c5d062e113482
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e8f8db8992e2e6003922c43dcc7e6b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d9fa403e72c01a0082c5d062e113482
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e8f8db8992e2e6003922c43dcc7e6b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d9fa403e72c01a0082c5d062e113482
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_9bc184f6d67993bfeef2587261632d87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4642, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4642, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e1d286e3e1d4338b3a7e28df6a38bd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bc184f6d67993bfeef2587261632d87
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e1d286e3e1d4338b3a7e28df6a38bd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bc184f6d67993bfeef2587261632d87
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e1d286e3e1d4338b3a7e28df6a38bd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bc184f6d67993bfeef2587261632d87
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e1d286e3e1d4338b3a7e28df6a38bd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bc184f6d67993bfeef2587261632d87
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ff35d5cb5813ba3df5559baca2937d4b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1047, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1047, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_028ac095d6614f2b361e4813b1231942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff35d5cb5813ba3df5559baca2937d4b
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_028ac095d6614f2b361e4813b1231942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff35d5cb5813ba3df5559baca2937d4b
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_028ac095d6614f2b361e4813b1231942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff35d5cb5813ba3df5559baca2937d4b
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_028ac095d6614f2b361e4813b1231942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff35d5cb5813ba3df5559baca2937d4b
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f4803ce22884e6eb5c6b62794d576d84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23bd35d07a8597402908f29ca134acd7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15261472761631012], [0.34060344099998474], [0.2098308652639389], [0.34933146834373474], [0.20875117182731628]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.28228238224983215], [0.4601689577102661], [0.23982450366020203], [0.4331777095794678], [0.05244790017604828]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_74e4bb58ef4bd58054cc65187579094e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23bd35d07a8597402908f29ca134acd7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05367068201303482], [0.36561113595962524], [0.4665828347206116], [0.2871648371219635], [0.07508329302072525]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.4062413275241852], [0.011468961834907532], [0.3539048135280609], [0.33214429020881653], [0.37686073780059814]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_30b740a3a14f7902b6b459bab9e4901d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23bd35d07a8597402908f29ca134acd7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12497831881046295], [0.2808596193790436], [0.11296632140874863], [0.20338411629199982], [0.2920495271682739]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.39957284927368164], [0.406246155500412], [0.24995972216129303], [0.3918718695640564], [0.4563639461994171]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_6d39ac8224b07ed9a162f1b311519e85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23bd35d07a8597402908f29ca134acd7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4368440806865692], [0.25516924262046814], [0.033184755593538284], [0.33212995529174805], [0.40790942311286926]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.47180286049842834], [0.31962135434150696], [0.32587990164756775], [0.07404229044914246], [0.4006950557231903]], dtype='float32').reshape([5, 1]),
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


class PrimitiveOp_3f0e790406440805cc480264271d48b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2359, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2359, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_56802fb84552b80e7d57cb86e404856f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f0e790406440805cc480264271d48b6
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56802fb84552b80e7d57cb86e404856f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f0e790406440805cc480264271d48b6
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56802fb84552b80e7d57cb86e404856f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f0e790406440805cc480264271d48b6
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56802fb84552b80e7d57cb86e404856f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f0e790406440805cc480264271d48b6
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b726d7f84f60cc8e7b19502c75b2fd26(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3049, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3049, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d85e0f9a5fcca16cae1272a514ae563(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b726d7f84f60cc8e7b19502c75b2fd26
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d85e0f9a5fcca16cae1272a514ae563(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b726d7f84f60cc8e7b19502c75b2fd26
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d85e0f9a5fcca16cae1272a514ae563(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b726d7f84f60cc8e7b19502c75b2fd26
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d85e0f9a5fcca16cae1272a514ae563(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b726d7f84f60cc8e7b19502c75b2fd26
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c316ecc2ee1fd7c7b424ee13f77a94d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3806, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3806, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c7986c25f1f9a36bad4507bb891efc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c316ecc2ee1fd7c7b424ee13f77a94d0
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c7986c25f1f9a36bad4507bb891efc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c316ecc2ee1fd7c7b424ee13f77a94d0
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c7986c25f1f9a36bad4507bb891efc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c316ecc2ee1fd7c7b424ee13f77a94d0
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c7986c25f1f9a36bad4507bb891efc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c316ecc2ee1fd7c7b424ee13f77a94d0
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_bd0338e0c6cec094334ae1bd17898a54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5cc331f0a0f34974d8726d249ced0cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0391080342233181], [0.06995806097984314], [0.4809706211090088], [0.22571998834609985]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.15882641077041626], [0.048668403178453445], [0.17359651625156403], [0.05119822919368744]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_332caca3d8ba9b0dbb2d0515db20b07b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5cc331f0a0f34974d8726d249ced0cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07258392870426178], [0.43579286336898804], [0.34903889894485474], [0.37684279680252075]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.43369781970977783], [0.18939900398254395], [0.3099403977394104], [0.4193747043609619]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_23c35128401d7f5e0ffce8ebc0c18273(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5cc331f0a0f34974d8726d249ced0cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20518168807029724], [0.4319309592247009], [0.47744256258010864], [0.42946767807006836]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.16607213020324707], [0.1927356868982315], [0.15653924643993378], [0.1810394525527954]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_9b0b896a61bcba2d152924a8427e98d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5cc331f0a0f34974d8726d249ced0cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.48733291029930115], [0.3396559953689575], [0.02164270356297493], [0.22589930891990662]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.05042451620101929], [0.23747287690639496], [0.43582987785339355], [0.027551813051104546]], dtype='float32').reshape([4, 1]),
        ]


class PrimitiveOp_ba17731c0f556016b8ad85e4ccf7c785(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2054, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2054, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4737b7522fb028005b406a2acd83b0af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba17731c0f556016b8ad85e4ccf7c785
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4737b7522fb028005b406a2acd83b0af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba17731c0f556016b8ad85e4ccf7c785
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4737b7522fb028005b406a2acd83b0af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba17731c0f556016b8ad85e4ccf7c785
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4737b7522fb028005b406a2acd83b0af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba17731c0f556016b8ad85e4ccf7c785
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_b37d58b4b5c6a4556f5d044ae4c9032e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4218, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4218, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4b4fce9f833b66d4f5c3479efa5731c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b37d58b4b5c6a4556f5d044ae4c9032e
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4b4fce9f833b66d4f5c3479efa5731c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b37d58b4b5c6a4556f5d044ae4c9032e
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4b4fce9f833b66d4f5c3479efa5731c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b37d58b4b5c6a4556f5d044ae4c9032e
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4b4fce9f833b66d4f5c3479efa5731c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b37d58b4b5c6a4556f5d044ae4c9032e
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
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