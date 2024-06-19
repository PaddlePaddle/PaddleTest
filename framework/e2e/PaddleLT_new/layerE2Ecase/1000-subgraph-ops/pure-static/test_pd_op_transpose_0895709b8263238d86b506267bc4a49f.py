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



class PrimitiveOp_004db2b2001a98539fe0d4e3ff5205f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 176, 176], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d7f16be4c0fe3250ab0a2361aa79901(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_004db2b2001a98539fe0d4e3ff5205f8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 176, 176], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7d33d3baffa3b7acf93705d6a28a01ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 88, 88], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8618425b01efb005422c32f48de3c3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d33d3baffa3b7acf93705d6a28a01ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 88, 88], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e1f4f5c5018e47ab79513c546f31a2ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 44, 44], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_375a5852c497423c66799432b1d589f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1f4f5c5018e47ab79513c546f31a2ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d7fb9b71ccc1e783eb5bdb89f855c0ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 22, 22], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_667bb9a25df3558df2890fee0bd6b50d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7fb9b71ccc1e783eb5bdb89f855c0ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fc5c2475ad006fd7c688fae8ff987215(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 11, 11], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9378859ac26e17deb0258fa04949f84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc5c2475ad006fd7c688fae8ff987215
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4160ece2f83e8707162143468dd7a564(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 176, 176], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09662c3081aea5e800833b7a3576bdb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4160ece2f83e8707162143468dd7a564
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 176, 176], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_696bad3fde869eabc719169e50d980ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 88, 88], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5767ca11ee781027691b0afa88a85cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_696bad3fde869eabc719169e50d980ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 88, 88], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5bff4200acbbf93c623f612fe7748cdd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 44, 44], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a556d482a5511a65fc8ebbf0d45e893a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bff4200acbbf93c623f612fe7748cdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 44, 44], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bf1c5183a962856090970855ec9bc386(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 22, 22], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d1f6fd14907599f77f061eb4da62384(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf1c5183a962856090970855ec9bc386
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 22, 22], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ec846601ee8f8466aa08c85f2871fe02(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 11, 11], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_13b9f8f33b4aca3357de4c49df62b779(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec846601ee8f8466aa08c85f2871fe02
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 11, 11], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c086dd764d41381070e45945c7e7225b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 184, 280], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f729c90781ac5617c0778a06c51cce4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c086dd764d41381070e45945c7e7225b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 184, 280], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_542218b6136c5e14ea5a11d78892e6c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 92, 140], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60f149d29b9ac87d918ea925f6236d90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_542218b6136c5e14ea5a11d78892e6c2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 140], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7eea9697ca6c6ee8807b2bc184f4fd0f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 46, 70], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a3842301ef62921bfa27ba922b42c684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eea9697ca6c6ee8807b2bc184f4fd0f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 70], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_42615672f64c2a7764af169f86f0acd7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 23, 35], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bb6a9523f1c7319107ea01aa00017a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42615672f64c2a7764af169f86f0acd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 35], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b25ecc1272b26e9dd05c968de9744613(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 12, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e36290d185b2a5cbb0820ca14362a11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b25ecc1272b26e9dd05c968de9744613
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 18], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fd3e66aaec9a8129a6ae26e440b2fb11(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 184, 280], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c3f32d14cc036caaddb8d367f5e0c51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd3e66aaec9a8129a6ae26e440b2fb11
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 184, 280], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d84aef348ae946ff65c2d60b916b1554(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 92, 140], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e76d5f9efc4299b8f558045eab7aa7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d84aef348ae946ff65c2d60b916b1554
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 92, 140], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7c8516c04338d4c648c7d9a1c2c25797(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 46, 70], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dcc8cba1e809080f39c3bf4bb7bcedbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c8516c04338d4c648c7d9a1c2c25797
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 46, 70], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_78f99e551210d2c7a338c837aeff36fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 23, 35], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53fe031eef2c38f89c57f4a9b8a66530(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78f99e551210d2c7a338c837aeff36fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 23, 35], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3b7304a101a8de5f7ec89f91f44c1006(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 12, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55f9d105d55eb2fcb17cffdf014619d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b7304a101a8de5f7ec89f91f44c1006
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 12, 18], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f9989e3907c452051a2730f12d482d69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 3600], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c988250545fc7d417a1dcf3717d4d855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9989e3907c452051a2730f12d482d69
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 3600], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3f80182cee7e4cff251e432c81bddca2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 3600], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_021d23358eb58c36ee3f9f30474ab73a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f80182cee7e4cff251e432c81bddca2
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 3600], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c69235fd3d31c63159f07f5ad042bf71(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 24, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c5876ad700e980ad6c027aa3a1d515a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c69235fd3d31c63159f07f5ad042bf71
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5fb31d094233c7b13fc7ef772520913e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 2, 24, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_14069fa497fe6b86febb04e5ae40660e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fb31d094233c7b13fc7ef772520913e
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c85fcaddcc667107ada757cc01b54fae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 24, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_281cd022c91dbc648010c8d5daebcbd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c85fcaddcc667107ada757cc01b54fae
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b4590d695db9f8983f1bc26345670654(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 100, 3, 4, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e3a4e13cd1072c830b56120be61e93d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4590d695db9f8983f1bc26345670654
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 3, 4, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7c6da0f0264d7892c1de7fb8d258e236(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 100, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee3ac876ea1956637e08da2ab2b76ef9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c6da0f0264d7892c1de7fb8d258e236
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_138bae9e0cfad9b1e553a4cf0b6144a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 100, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f2f0176f9eeb17731d951249ed3b46e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_138bae9e0cfad9b1e553a4cf0b6144a3
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ca161ce2677a5dd4733c58d95a2ca851(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 8, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6516e33c2491b8c3401c3e89b0540c7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca161ce2677a5dd4733c58d95a2ca851
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 8, 8], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e619824f41249824280889dd4e6db702(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 15, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6d4a326d2f9474a5feb5689a8aaacaf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e619824f41249824280889dd4e6db702
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7fd087e9b8d337f4e1b6470e74a6e682(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cb6ca6b66b09ef0568b67b46a7eb4854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fd087e9b8d337f4e1b6470e74a6e682
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4a65e69570ac7936e6fe370c04384258(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 91, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c6b6c0d3a94c2a198ef93a6da959487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a65e69570ac7936e6fe370c04384258
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bcb65fe6d53f3d62ff76d7cbac56b040(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 4, 7, 4, 7, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e9f39674b18afdba3edc11ca2a32083(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb65fe6d53f3d62ff76d7cbac56b040
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 7, 4, 7, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_329d5670637f8070db71c773aa95c706(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 16, 49, 3, 6, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f5e9cb8f3cc1a18fb17ce38f096c2e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_329d5670637f8070db71c773aa95c706
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 49, 3, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3412bda6865389dff7f108ee1309863f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 16, 6, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b9f9d3c072b3ed0223136d2f9d33126(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3412bda6865389dff7f108ee1309863f
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4dad0e76957d256859563cd800db89cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 400], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5ac85a410e97f8f79b85c60dcf97b0f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dad0e76957d256859563cd800db89cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 400], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6c51269703391cb3c9db022ed4a74d00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 400], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_207b632120344fd362c42eaadfea9f4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c51269703391cb3c9db022ed4a74d00
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 400], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e9bb38c29a78dd6b9d08b9b432d8c815(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 196], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f4b92d329d4810bc91393a50386b649(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9bb38c29a78dd6b9d08b9b432d8c815
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 196], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_31adf99fcd3dc5d1ee65c307e60599d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 196], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf415c498cbed0506f631419fd01ece7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31adf99fcd3dc5d1ee65c307e60599d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 196], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0d414373489457c5158a516adb691674(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 3, 12, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_14e687af42f81c0bb5310493e2462e54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d414373489457c5158a516adb691674
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 3, 12, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b439e2092c7de8d7c3138eff0f569baf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1025, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a8b880b16bbc822f98459cd832df383a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b439e2092c7de8d7c3138eff0f569baf
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cfd15314c427d1dff979b48b33d8a256(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1025, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_42cf8648be880b647e4b2a1697772ef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfd15314c427d1dff979b48b33d8a256
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5e9f39674b18afdba3edc11ca2a32083(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb65fe6d53f3d62ff76d7cbac56b040
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 7, 4, 7, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0f5e9cb8f3cc1a18fb17ce38f096c2e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_329d5670637f8070db71c773aa95c706
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 49, 3, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7b9f9d3c072b3ed0223136d2f9d33126(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3412bda6865389dff7f108ee1309863f
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fb85ac39ae2f603bb1d17b3050053cd5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 4, 7, 4, 7, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f0fbf3a6ea373e522184c784bbcc0df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb85ac39ae2f603bb1d17b3050053cd5
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 7, 4, 7, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4e09661471d5d54e80c75cb6b070a2ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 16, 49, 3, 6, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_96f188ef1ac4ac506aec34675733b2f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e09661471d5d54e80c75cb6b070a2ed
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 49, 3, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_dad816d24251c1ac6c20aecd26b8083a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 16, 6, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_83326279fcf618dbe761c47360d965b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad816d24251c1ac6c20aecd26b8083a
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7903acec907903bca894e8f134486b89(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_725ba409e7d27b90fd3ece82e7cdab51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7903acec907903bca894e8f134486b89
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 8, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2c71de12601557fccc010dfc3e979eee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 2, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_531c7f2e4a429324623dc31e2762b891(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c71de12601557fccc010dfc3e979eee
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 8, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9f6e9a1a605b343f4fb73ceb3d2681d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_593bcce33a37043244a4cc3062784089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f6e9a1a605b343f4fb73ceb3d2681d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_02bc6a38dccb8646403b48d70f31e5be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70e56f911a3d058653b341626d565885(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02bc6a38dccb8646403b48d70f31e5be
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b894984190ad9dcad0250e238146175a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 232, 16, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f8f635178f93c4e3662123f776c2b212(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b894984190ad9dcad0250e238146175a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 232, 16, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e2420ed4590915e918010772c890c36c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 256, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_27fa2639720dc55b890ffc91f03b1c74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2420ed4590915e918010772c890c36c
    def get_inputs(self):
        return [
            paddle.uniform([100, 256, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9cecf660b596c07ef2e1ad1015291353(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ac4aa45aa4c988f74a952dc10582a828(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9cecf660b596c07ef2e1ad1015291353
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_06f827971a8f84f612d65cc80006e776(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3145fc8dcf54cc47a4c580d50f364469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06f827971a8f84f612d65cc80006e776
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_41ee67b03399b8077d7c6336443a4403(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3bdd7918d3778d5b4a6d8f2352f526c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41ee67b03399b8077d7c6336443a4403
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_50e179c887d95e472bab06e86b01d414(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 196, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_547ca365aa33ec018d8aaabadc95f9f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50e179c887d95e472bab06e86b01d414
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_43eede5085a2c790494760fca87050a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 384, 196], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6334af855f94b146974a510009f000ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43eede5085a2c790494760fca87050a1
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 196], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7ea56368bb6e235ed3c1867df98613fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 2116], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1421270bb6cbfe976860bccedb3cbc6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ea56368bb6e235ed3c1867df98613fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2116], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fb542bc2349354f902744e4524524f51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 2116], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_088d398c2e31173abda2b62048c2fcf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb542bc2349354f902744e4524524f51
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2116], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_610831900cc4cf581a3b2e9fbeca972b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 197, 3, 3, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b9a8d1b643c00a8314a5ecdd053e6431(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_610831900cc4cf581a3b2e9fbeca972b
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 3, 3, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cb146fd3cb42137c92739798fb7bde42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_45ddddba7fe280d6a722a5e93fb03057(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb146fd3cb42137c92739798fb7bde42
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b70ec08e41922dc9bca2a540adbd9f52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_726846785631b8f45172469dd57b446d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b70ec08e41922dc9bca2a540adbd9f52
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_df87dbc8e265819adc369d8222e2dc5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d8dcdf543a7421f3cfb95ae1e0dcd78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df87dbc8e265819adc369d8222e2dc5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_484c238e6df29f547f33f268e9fb6ff3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 196, 4, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e74c54a2185c24894ca57757c0ae4217(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_484c238e6df29f547f33f268e9fb6ff3
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 4, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e74c54a2185c24894ca57757c0ae4217(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_484c238e6df29f547f33f268e9fb6ff3
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 4, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_35f38826c3b6c1a7e66cdd0e52b9ee65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 196, 4, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_884905c1446595619a911b4ff6b5ab3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35f38826c3b6c1a7e66cdd0e52b9ee65
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 4, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c9b966bedff7c3269c23d09cfb084f0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 4, 196, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cb8541021deaa755ea7c9bb62db6e61e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9b966bedff7c3269c23d09cfb084f0b
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 196, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_18b8d9a39fb893fb184e100b59404a17(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 196], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59bc088890b45d38d7f024b538c535f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18b8d9a39fb893fb184e100b59404a17
    def get_inputs(self):
        return [
            paddle.uniform([4, 196], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_647082ef376f5a73c0964f696d448383(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[38416, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a74646311d97a4ef051fc48e219382cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_647082ef376f5a73c0964f696d448383
    def get_inputs(self):
        return [
            paddle.uniform([38416, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9874f711042e2bfd107011a52d643e65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 15, 16384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a7d1b0268d349ee63c28b2a6f5f09263(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9874f711042e2bfd107011a52d643e65
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_736429e2d6de83812117a67bd8bf0eec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 16384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_91ce96f4f675c6b3875a1bad1955a935(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_736429e2d6de83812117a67bd8bf0eec
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1c9267bec5ee55f9afae9c6d978b6d09(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 91, 16384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4884a40b93594d469e258132bdbdd56f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c9267bec5ee55f9afae9c6d978b6d09
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7f354d453a96c3212e0e63fbfbec637d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 136, 208], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b24a112d5ed901cca154b600657f621e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f354d453a96c3212e0e63fbfbec637d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 136, 208], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_65e8a94562e24f9b760483f6677fdd22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 68, 104], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d65066e1ea9302926e6d19fd26832a76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65e8a94562e24f9b760483f6677fdd22
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 68, 104], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e1e37094676421b43042f70c28c7d013(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 34, 52], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8fcec4080aa031423884ee2bb6bde541(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1e37094676421b43042f70c28c7d013
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 34, 52], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c8ff8def20d256b50ebd7230895f68c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 17, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3de917fb4eb751b44fd9479147e4b8d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8ff8def20d256b50ebd7230895f68c7
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 17, 26], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_011a7d3243ea07f736899fe8eac5cc73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 9, 13], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_484ea649748c0cf60458d835d9b18237(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_011a7d3243ea07f736899fe8eac5cc73
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 9, 13], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8e7c249acc73fb35ef843c5eb6e18cac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 136, 208], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee0df2eba112e6e28542d822b6b2600c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e7c249acc73fb35ef843c5eb6e18cac
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 136, 208], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_838dd935d889cf5b1aacf0fdf3895208(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 68, 104], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59e8dbc6ad8daa1b272166aff1636b92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_838dd935d889cf5b1aacf0fdf3895208
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 68, 104], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c50dcca807362ab8d230bfa169092ab2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 34, 52], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d853bc1f70c4ef2d20e663721b58ffb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c50dcca807362ab8d230bfa169092ab2
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 34, 52], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_658c940c86e5edf0b7a1c42d3ea533e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 17, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b94954540000f4cd97e012996f2d14f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_658c940c86e5edf0b7a1c42d3ea533e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 17, 26], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bf41a6cf07803d7a86adfcb894f91b32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 9, 13], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_514ad0beef82cb90fac3c83404b5199c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf41a6cf07803d7a86adfcb894f91b32
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 9, 13], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_540e613b338922b91701499f99b9e168(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 6400], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e7a7e5b4e1ac764f7cc52f270e65538(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_540e613b338922b91701499f99b9e168
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 6400], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1df100311b9b31b9092e2e1bc53cc5b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 6400], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a67cecf7b928d01be94b40302bf801c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df100311b9b31b9092e2e1bc53cc5b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6400], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c151f3cdc2c5d99745e9316d94075701(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 6400], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d7ecd600f9f8d5a06e794af6aa654e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c151f3cdc2c5d99745e9316d94075701
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 6400], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0b81e24bb92111aa18dc7e4b62ed85c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 2, 7, 2, 7, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a30c84ee0173b7456f5b19d882e9111(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b81e24bb92111aa18dc7e4b62ed85c2
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 7, 2, 7, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_58d46d6af14fde8039b960580dddc065(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 4, 49, 3, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9d1527194c9f118e732d746c54716b6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58d46d6af14fde8039b960580dddc065
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 49, 3, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_35b94e436c33d86a3020111bc76e2ddd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 4, 12, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e8944027eefcf98155844dd21093a22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35b94e436c33d86a3020111bc76e2ddd
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8ce519591fb11496bbf51928e75c4ccb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1ccbdb7879c9793b76fe64d857256b77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ce519591fb11496bbf51928e75c4ccb
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bfd3f30f3ea72d2295750820bd79414d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 2, 7, 2, 7, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_646a62355a4f033c940be394e1e40307(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfd3f30f3ea72d2295750820bd79414d
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 7, 2, 7, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4823e8f104c33c1b0f5551ed26ca94bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 4, 49, 3, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_191b83046528c7e93a8100ee2c26b597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4823e8f104c33c1b0f5551ed26ca94bb
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 49, 3, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0487eb4fba7cdca9eae703b7ccacb830(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 4, 12, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a4392384ea9afcb19ad621a79cec5a45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0487eb4fba7cdca9eae703b7ccacb830
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_25224b976a4c62f9b73c99177728f133(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 38, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d8d959f7f51d5a5042a304e77e7de73c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25224b976a4c62f9b73c99177728f133
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 38, 38], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a7d0a429d7247dd133d432fd00638fad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 84, 38, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_62a740907a4a84f7f58c260e57beaab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d0a429d7247dd133d432fd00638fad
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 38, 38], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b718b8fdbb7a99e8261977231a5b74b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 24, 19, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_207f6046dfab1dd2e602d5e24b5d84a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b718b8fdbb7a99e8261977231a5b74b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 19, 19], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_53349c9886a0c960a5920f0066ab9a74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 126, 19, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_deb39319f286c48daaede0d9b052d3dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53349c9886a0c960a5920f0066ab9a74
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 19, 19], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_90ed91c7719afdfa3eb9e41c06bb7e64(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 24, 10, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b03667c64d90ba5cef86ad1d04ff7b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90ed91c7719afdfa3eb9e41c06bb7e64
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 10, 10], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ad71017a058003c28d0a82342c6e0fc4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 126, 10, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9225da26a7a1efc2117f39366063a89d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad71017a058003c28d0a82342c6e0fc4
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 10, 10], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cbd6962fbf999360ca24f7d39466e6ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 24, 5, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b25ee8ba44fdd33bc055494bdc6bab4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbd6962fbf999360ca24f7d39466e6ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 5, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6fec08892a6349a5f468a4207e7cd26e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 126, 5, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f5bad14478b75df149bfe6e62ea1565e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fec08892a6349a5f468a4207e7cd26e
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 5, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_739eb0dbf3d157771d0082621e573fe8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f736d5a65d007d70284cf0a5e5863828(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_739eb0dbf3d157771d0082621e573fe8
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7d43255a16ed6a9bfdf93993eb6a5fda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 84, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c20906aea8be85be1d387e36728ac431(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d43255a16ed6a9bfdf93993eb6a5fda
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f3e4371bb12f62f093ca646b7900bae8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c14f419bb4fd9a1795458d8f763f1cda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3e4371bb12f62f093ca646b7900bae8
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.02558143436908722]], [[-1.3986389636993408]], [[0.04337681084871292]], [[-1.131929874420166]], [[-0.6939791440963745]], [[-1.4564203023910522]], [[0.6139976382255554]], [[-1.8328877687454224]], [[0.28446322679519653]], [[-0.8854150176048279]], [[-0.33872050046920776]], [[0.37590935826301575]], [[0.15059220790863037]], [[-0.5969091653823853]], [[-0.0749569684267044]], [[3.261270046234131]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class PrimitiveOp_1ab02cff6dfda5a85b26a6b1aaa0c9a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 84, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8716de4dc282812511b44d9c8b425b7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ab02cff6dfda5a85b26a6b1aaa0c9a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0264b42ee0b22b0800ba3f737b6b7897(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 784, 6, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49aa8180a927d3b40e93d57307a03b77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0264b42ee0b22b0800ba3f737b6b7897
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_76dee8755d5dd20cf8c34a12ce9eb7c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 784, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b596c1d8d87e902dcbe04cfad547765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76dee8755d5dd20cf8c34a12ce9eb7c2
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_60cb79dee0cb1e9fe788d25e98edb957(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 192, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b112e07e24617a18684eba67aef9c029(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60cb79dee0cb1e9fe788d25e98edb957
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_960222d2177659dd07d97e15a0dc249b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 2, 6, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b6efe8292f6757a2cf439a6d321532d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_960222d2177659dd07d97e15a0dc249b
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8718f08430db77a6af404763fce7563f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 6, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0549042d52a25b96aa5bb8b6328d7e96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8718f08430db77a6af404763fce7563f
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a9a98da78b84aaa182a7264c6f891387(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[960, 2, 96, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d082daebb7a77d02e5557c4c0641566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9a98da78b84aaa182a7264c6f891387
    def get_inputs(self):
        return [
            paddle.uniform([960, 2, 96, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_40bd8253c7a8271dd1fecc8f502ce64a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 96, 1, 1, 96, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1fab766b777ddd1e886c36e0ee9a7c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40bd8253c7a8271dd1fecc8f502ce64a
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 1, 1, 96, 48], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0c1527eec68163910f412fc52da11d63(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 1600], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1264d0e662ce353c9eb4552bdf62c982(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c1527eec68163910f412fc52da11d63
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1600], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_dc4e16bdca22ccda1ee9297bb7762fd5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 1600], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_62fd29f80e8d31a187a735939a6d8016(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc4e16bdca22ccda1ee9297bb7762fd5
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1600], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d8090b6a7507ca030d8a62153e74211b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 120, 216], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b5a6510fb02582161fd38e56e7fbcb2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8090b6a7507ca030d8a62153e74211b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 120, 216], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_dfbe3a6aa43880fbf6cebecb511fe828(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 640, 3, 2, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fb571c2a6488ae4bb4c8be5cb8bcc989(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dfbe3a6aa43880fbf6cebecb511fe828
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 3, 2, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_621f81f1cbecca85acf62772f42769cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 640, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1d6b450ce981e753f97173384b457f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_621f81f1cbecca85acf62772f42769cf
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6f363264539178be5ec5a91ba6de5e6b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 640, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8611b15ecdf528c3a0babeac3a837ab9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f363264539178be5ec5a91ba6de5e6b
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_49aa8180a927d3b40e93d57307a03b77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0264b42ee0b22b0800ba3f737b6b7897
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8b596c1d8d87e902dcbe04cfad547765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76dee8755d5dd20cf8c34a12ce9eb7c2
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b112e07e24617a18684eba67aef9c029(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60cb79dee0cb1e9fe788d25e98edb957
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b6efe8292f6757a2cf439a6d321532d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_960222d2177659dd07d97e15a0dc249b
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0549042d52a25b96aa5bb8b6328d7e96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8718f08430db77a6af404763fce7563f
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d24e171513e925cbc24a65ec586ad6c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[144, 4, 96, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4954e57716a35b16ec0488722d8e02d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d24e171513e925cbc24a65ec586ad6c7
    def get_inputs(self):
        return [
            paddle.uniform([144, 4, 96, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5a4da4b96353afe277f9325bb56f77d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1, 24, 48, 2, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97b383ef66a21bab295358d62b476ec6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a4da4b96353afe277f9325bb56f77d9
    def get_inputs(self):
        return [
            paddle.uniform([6, 1, 24, 48, 2, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ef4be89125fd8bfec5dde5279597f614(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 8, 7, 8, 7, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_902b16f142e51167681ae99ebfeb8098(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4be89125fd8bfec5dde5279597f614
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 7, 8, 7, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c09db675753c27b512b2ff6d36a883ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 64, 49, 3, 3, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_16425d7375cb8d5b4c9d3b8f861422fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c09db675753c27b512b2ff6d36a883ef
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 49, 3, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a821b35effdac8f38776368a78c6e04e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 64, 3, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_497c6edf24c50507c6ae0d8d4787994f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a821b35effdac8f38776368a78c6e04e
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e7c031aaaac960f4b4a468a979139505(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 16384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_946b51c66731e361f066ebe8d2db942f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7c031aaaac960f4b4a468a979139505
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3a12c3fdb2e6599927733e10ca561501(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_edb1f4b1dc3f60d186e1336a7db4249b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a12c3fdb2e6599927733e10ca561501
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f735e5cf39f651171a1eeb829d84cd1e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 15, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a437e8b085a001367d484c88e8def61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f735e5cf39f651171a1eeb829d84cd1e
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_51f6e4e493661b6052e041f3ff5dbbbc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 96, 3136], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_105258f83ab58884856acffda56f7f0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51f6e4e493661b6052e041f3ff5dbbbc
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 3136], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cb1fe9c9c86b255a4d69a38bbf6719b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 200, 3, 2, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ad9657bd213553b4b40296f42cff1af3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb1fe9c9c86b255a4d69a38bbf6719b8
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 3, 2, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_20ae0c1569c8c951b6f52ba05255c8da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 200, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e123f5e671278fda833009bac0e27a6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20ae0c1569c8c951b6f52ba05255c8da
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9a68846c5d3935bfd0061e1e16314a27(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 200, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a6696530818cc21acb88261fddb3342d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a68846c5d3935bfd0061e1e16314a27
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d801a3f63884f9364bf925bb56c70c27(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 196, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_03782e4d944f1e61a5e270d8dad34919(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d801a3f63884f9364bf925bb56c70c27
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3c0688ceb8a47d57e27f99c90a19bb9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 196, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e3e8f1d92a72bcdcf50c2fd7f82d9ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c0688ceb8a47d57e27f99c90a19bb9b
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5dff5785c503eefcb9c9b84253e71133(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 384, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f9dc8d739d8704d8b4a910ccfe9cfd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dff5785c503eefcb9c9b84253e71133
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7f8042f2a9c21448974db77e26270663(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 2, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_03a63373c6e78cbf2b3e106039da6c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f8042f2a9c21448974db77e26270663
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_12174faa73e86c03464ec8a20a68da9e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 12, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_01bf2b73fe61c9592942475a783ed138(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12174faa73e86c03464ec8a20a68da9e
    def get_inputs(self):
        return [
            paddle.uniform([43, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_581351d2c9f222d4e806d2f588dfd829(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a84564e60d025ac9b5b2ab5397fd4962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_581351d2c9f222d4e806d2f588dfd829
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e6f4128b5aa91d80541ab4f5fcb6f9e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e84af7c453015be1be4a52c10edbf23f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6f4128b5aa91d80541ab4f5fcb6f9e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fc060999287e06fe5e62ec51a3a39d4e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60d2b9b127302e68ad1d9d0fd614ef7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc060999287e06fe5e62ec51a3a39d4e
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4f6c6361dc5654339d27ac533307ae9c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 169], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6a075d50ad9c4bf912cb6774e8c1b7b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f6c6361dc5654339d27ac533307ae9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 169], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_398e3e4874eca6a3306119ffa4329d14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 169], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1da26df5301b255b23ca5b8627d7e394(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_398e3e4874eca6a3306119ffa4329d14
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 169], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f20086c37f41b11be18424d31e4e562f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8fa0746e38592a34f43f700b658ba95a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f20086c37f41b11be18424d31e4e562f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_112b2548816103019986f69677d7b728(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4735edb7e2f7b773ae387acb8f6d91dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_112b2548816103019986f69677d7b728
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3c04ce9e038dd692713a55140cd80211(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_01ef0aacdec361afc21c5b6884a93840(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c04ce9e038dd692713a55140cd80211
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f70dc5608adc55511e8e76930a187822(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23a22551c0e41444e28cd486276d2685(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f70dc5608adc55511e8e76930a187822
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_40e713acc4060b0b8cbf4973a1cc0b39(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 9216], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e578ebe5b26367a0d3164996dadf957e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40e713acc4060b0b8cbf4973a1cc0b39
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 9216], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e9c57a07353d6475f7774238eb7219a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 9216], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_be8dc664841d625f341eec8ad6226a55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9c57a07353d6475f7774238eb7219a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 9216], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3044f5675935f19c9e65eaa393ecf842(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 9216], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_04138d9a485143b64efa8eb5ce4baaee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3044f5675935f19c9e65eaa393ecf842
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9216], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ef6eca8324cb89c425ead0be71e45bdd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 784, 6, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e1597e3f9acaefacfa1887f9a06b894(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef6eca8324cb89c425ead0be71e45bdd
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_eb9f70282899345810179c07997ffef4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 784, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_24fa89f9e243f08ae2c8f4a3926f4c00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb9f70282899345810179c07997ffef4
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4dc2635fc07050ef006a0bdd589c884e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 192, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_232ad9e18a32d2feca61f4f85ae4050b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dc2635fc07050ef006a0bdd589c884e
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2945ca61ad8e249c717d30897e1fd40f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 2, 6, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0c306ebccb1028ac50fd08ab4b199814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2945ca61ad8e249c717d30897e1fd40f
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_765dea6d119b72d25e404af98eb9718a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 6, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4cd76f5ede96d6363cd47d42235b7000(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_765dea6d119b72d25e404af98eb9718a
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3a6321911f0ab1f806208e5a85cef5b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[384, 2, 96, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_750f9a652817c4b00ac7ea69b8d39fae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a6321911f0ab1f806208e5a85cef5b8
    def get_inputs(self):
        return [
            paddle.uniform([384, 2, 96, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_66ba65cd188fb55fcc87ad08b171a477(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1, 96, 96, 1, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b3b8b0d7b73eb8c796d59d985a6d1724(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66ba65cd188fb55fcc87ad08b171a477
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 96, 96, 1, 48], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5e3e8f1d92a72bcdcf50c2fd7f82d9ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c0688ceb8a47d57e27f99c90a19bb9b
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b93c975b04c598aa3c1e7ff8a4a2939c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 384, 196], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d0685b45b469eaad5c219d388ef91aa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93c975b04c598aa3c1e7ff8a4a2939c
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 196], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c489af8772de81a0cb76656183cef816(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 60800], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_15b7550d0f692e8509dea47c5bd9315a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c489af8772de81a0cb76656183cef816
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 60800], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b399c67bb568f1d6cc4b530cab89b374(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 60800, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d072feb2feeddf5b4f319eac5730650f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b399c67bb568f1d6cc4b530cab89b374
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_15b7550d0f692e8509dea47c5bd9315a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c489af8772de81a0cb76656183cef816
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 60800], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ded1a7c05af155087ddb75b3ab878e1d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 3136, 3, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_769ebe0f31eee0529acfc9122ef40172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ded1a7c05af155087ddb75b3ab878e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_861b55d23659bfb7e30bb07acda5fed3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 3136, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4aada27a760f71f394e5e9c393d740c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_861b55d23659bfb7e30bb07acda5fed3
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b1c5ea8fe5310ea4b3cb7a0cda27ec5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 96, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55622116095b67cd7a8605c3ea883674(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1c5ea8fe5310ea4b3cb7a0cda27ec5d
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_db74162f4e6a34eb35457836172edf90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 2, 3, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_660bcef112fbf7db74c11e7373b8a1d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db74162f4e6a34eb35457836172edf90
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1c21259e70f761ede775e12e07f6f854(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 3, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_29245447b95c1da982dd9b88e75b7253(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21259e70f761ede775e12e07f6f854
    def get_inputs(self):
        return [
            paddle.uniform([11, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_01569d6d711e090c69926ab2c745c449(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97606e7f3e8f0b81cce0a17d1059d90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01569d6d711e090c69926ab2c745c449
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6e1bfc6372783f01f98ed165cbe5693c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 192, 784], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5bc0e9cc69a43f17541e794c2106bbf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e1bfc6372783f01f98ed165cbe5693c
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 784], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4e2f5bea49ff89ad332d2a3e44432b5f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4312, 16, 2, 4, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d458f1f41432200373c06dd9a0922fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e2f5bea49ff89ad332d2a3e44432b5f
    def get_inputs(self):
        return [
            paddle.uniform([4312, 16, 2, 4, 6], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_134f860ea1405df8ba3d6cf28c13342a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4312, 16, 4, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f52d1b6291fb2b73c59f0ecbe42e8a38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134f860ea1405df8ba3d6cf28c13342a
    def get_inputs(self):
        return [
            paddle.uniform([4312, 16, 4, 6], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_24df84a92a70abf73ccdca04b0d3bfe9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4312, 4, 16, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7da7dbf768a5ea3695c5cfe86e16f22a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24df84a92a70abf73ccdca04b0d3bfe9
    def get_inputs(self):
        return [
            paddle.uniform([4312, 4, 16, 6], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0ef7f11bc4e7114e279e1f498e29fdd6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 96, 9216], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6a4dacf2228390cc08946f6afc2bca6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ef7f11bc4e7114e279e1f498e29fdd6
    def get_inputs(self):
        return [
            paddle.uniform([6, 96, 9216], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d33ebe7eb2350c6a5f7fbef046cdb950(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 160, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9dd939c1629088c18530b0250984b12e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d33ebe7eb2350c6a5f7fbef046cdb950
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 160, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3b2b3755fd6cbef82e8e4c40e1d79cc1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 80, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc290aa5e246c215fb75cd93840f2fbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b2b3755fd6cbef82e8e4c40e1d79cc1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 120], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_025388fd9e00346610a5a8d6bf4e6005(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 40, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55bae06fbffb2ea4bd0d2f12a733368a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_025388fd9e00346610a5a8d6bf4e6005
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7b4f5c90c6fd97c7072322c3f159a1cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 20, 30], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_52784c63c8ef208b3347622deec87aad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b4f5c90c6fd97c7072322c3f159a1cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 30], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_be85d04100b559b512b10ddee033a188(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 10, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6fe61964492db54ebade255a64106f8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be85d04100b559b512b10ddee033a188
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 10, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_41fe3a6483ff0ad187409d1b1e6f5484(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 160, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_25e5c87afbad093bd5c7338c77cc1810(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41fe3a6483ff0ad187409d1b1e6f5484
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 160, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_03ff2fa85203d463c5bbf5e746cd5428(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 80, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0a54e6c27441671618ada488b3afa87c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03ff2fa85203d463c5bbf5e746cd5428
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 80, 120], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_914169f7edb2ee922ddf52d5f3c38163(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 40, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3277388aba4359c6bd6737c96d2a8787(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_914169f7edb2ee922ddf52d5f3c38163
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 40, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cce98897e476af4be17fc26091f3ba1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 20, 30], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b5470f17e18b813ede7b80c0eb8292b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cce98897e476af4be17fc26091f3ba1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 20, 30], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_799b315ebdce0326635ae59331852079(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 10, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_89297c44b7fde54cc29c913aaf9e8503(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_799b315ebdce0326635ae59331852079
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 10, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6334af855f94b146974a510009f000ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43eede5085a2c790494760fca87050a1
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 196], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_513a7aa4a27595b824e8e952e1bb0c0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 2704], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_133643707e5a603060a38f0e069607b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_513a7aa4a27595b824e8e952e1bb0c0b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2704], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e25ce0e126b680a08732401d6e34c41d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 2704], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9813a90a013b07c2d69599597a4e0186(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e25ce0e126b680a08732401d6e34c41d
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2704], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d9c3194fb9648928281c02df6c81aca6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 768, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b23f30da9c7e03e77b63958f81b8e041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9c3194fb9648928281c02df6c81aca6
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c61d9a0d4783603720a717a2cec2850e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 152, 272], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9a9bda9f53c88b1f5c1f53fd7c9e6569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c61d9a0d4783603720a717a2cec2850e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 152, 272], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b1d629783b007af804fbe95a73f06c4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 7, 1, 7, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7834d5bcee60fe411bf1842c4d972fe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1d629783b007af804fbe95a73f06c4c
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 7, 1, 7, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_77a7e7996acbc6fcd97f4e0f2af9d257(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 49, 3, 24, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6fd30380c37e45a892ad4738bb059e25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77a7e7996acbc6fcd97f4e0f2af9d257
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 49, 3, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5ab7b62f825fe7beca19a9aa43990acc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 24, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_491ce757e507103a23f7469d0554e25e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ab7b62f825fe7beca19a9aa43990acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_343f7d8725b39b02a9cffe0ced47c44c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 676], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_051734f99c6c33b33858c391dfbcc053(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_343f7d8725b39b02a9cffe0ced47c44c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 676], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_341bbfa11050d0985660b1721a2a2775(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 676], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cac556089bd23fb787c69455bf8bf970(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_341bbfa11050d0985660b1721a2a2775
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 676], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_82aa51e07a4573708aef60fcd33c2833(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 289], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_88c8fdbde53013d6879a05b32e43f9a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82aa51e07a4573708aef60fcd33c2833
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 289], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a2c102b3ccf6d6de8521fd7d4ef7a611(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 289], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f2b4e27cade1386f8dfe1ef2fa516306(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c102b3ccf6d6de8521fd7d4ef7a611
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 289], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a7d1b0268d349ee63c28b2a6f5f09263(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9874f711042e2bfd107011a52d643e65
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f29d61a42275d6a4289b8be3acc810c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 15, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_76673f5e72e3668402cdedc421ee1849(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29d61a42275d6a4289b8be3acc810c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3145fc8dcf54cc47a4c580d50f364469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06f827971a8f84f612d65cc80006e776
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d0617dcf33c4fe3b1979074e17e5dbbd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 91, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60def92137baf16976c789140647063e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0617dcf33c4fe3b1979074e17e5dbbd
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1021a233ed67b23276c8f0415e98e21d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 256, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c895a33a114b2f4e9d1e615ad7396b1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1021a233ed67b23276c8f0415e98e21d
    def get_inputs(self):
        return [
            paddle.uniform([300, 256, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2c5019a283b003b024cc7047402fc7dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 16384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe516bc88d1e62ad909967ac364c6161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c5019a283b003b024cc7047402fc7dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6ab9878002c9d8f8f9432f54b7fc4017(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 96, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_915c996639eb18129ad35c73851d319d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ab9878002c9d8f8f9432f54b7fc4017
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 40], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_55cc5bba00b9cb3fdc64e55f67d86fe8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ce79ba9729dde7624b9d55bc2fa90a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55cc5bba00b9cb3fdc64e55f67d86fe8
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_90ca448d35f938a13e0e478d14020e43(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 160, 3, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_03ed7e5f504db3f0180f58cb95e28a1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90ca448d35f938a13e0e478d14020e43
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 3, 8, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_761b85a9689b58fa781b6ec54cd18a95(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 160, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0846c88f24f30219065679c9a5ca4333(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_761b85a9689b58fa781b6ec54cd18a95
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cda03d70089208e085bddc13789dc777(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 160, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a84da3244b3e9625fda1a7032ccf74b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cda03d70089208e085bddc13789dc777
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0c0adc21bd253225d1aa215ee7d0cb63(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 72, 14, 25], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e3ced91a03d757ae0270299e411c533d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c0adc21bd253225d1aa215ee7d0cb63
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 72, 14, 25], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3c949dac7a58daf808d01015f8193749(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 529], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ea7097313e81d30e08ff4d7f68c5b138(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c949dac7a58daf808d01015f8193749
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 529], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d205d0e5c4f3c6e2149dfbcce0202fe2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 529], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b042e29ff27466a42236419c539ff97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d205d0e5c4f3c6e2149dfbcce0202fe2
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 529], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c895a33a114b2f4e9d1e615ad7396b1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1021a233ed67b23276c8f0415e98e21d
    def get_inputs(self):
        return [
            paddle.uniform([300, 256, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4aada27a760f71f394e5e9c393d740c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_861b55d23659bfb7e30bb07acda5fed3
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_407e86043743db80b886354782c8f9d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 96, 3136], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d025e6d41ed33c1cf5f5392c03e134f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_407e86043743db80b886354782c8f9d0
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 3136], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f0fbf3a6ea373e522184c784bbcc0df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb85ac39ae2f603bb1d17b3050053cd5
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 7, 4, 7, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_96f188ef1ac4ac506aec34675733b2f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e09661471d5d54e80c75cb6b070a2ed
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 49, 3, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83326279fcf618dbe761c47360d965b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad816d24251c1ac6c20aecd26b8083a
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4735edb7e2f7b773ae387acb8f6d91dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_112b2548816103019986f69677d7b728
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_291f7e7c973ea004a299570d6cd22ca0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c8a8f3cd33c547b0318ba04501b1b5d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291f7e7c973ea004a299570d6cd22ca0
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f4a3f80c348ab2536c1285fdd1064a90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 3136], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a623b41c0bc5785a1e2c88d6cb5e85c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a3f80c348ab2536c1285fdd1064a90
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 3136], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_93fff2ee76cb12b87038ad024d7858eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 3136], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f8cc7fe59e5958b16e5282f46bd47298(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93fff2ee76cb12b87038ad024d7858eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 3136], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a18a2355f862818d6ad8b6fe7b9636dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 2, 1, 12, 24, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc34476197396feb05f58aba30a460b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a18a2355f862818d6ad8b6fe7b9636dc
    def get_inputs(self):
        return [
            paddle.uniform([6, 2, 1, 12, 24, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bfb86674f08e2524dd57ca19edcc36a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 4624], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_382842bd160162eddda7d7a178127394(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfb86674f08e2524dd57ca19edcc36a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4624], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_098eaf75e3bce05255ce6aadc66086b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 4624], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00cdcc73b41b3e92582986f830225ec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_098eaf75e3bce05255ce6aadc66086b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4624], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_79ce896732e27d96b289025f8096e116(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 4624], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_72776645524109504ac966d784145000(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79ce896732e27d96b289025f8096e116
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4624], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a4807b5e9d46cce74dc40e3066bd70d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, 32768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6523bf7f2ab3e87ed97307e1c0dbe1d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4807b5e9d46cce74dc40e3066bd70d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_595f21ae57133de93b9d170c3981407a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 320, 3, 4, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_834a01a99007e20f7d8b618a2f8e26f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_595f21ae57133de93b9d170c3981407a
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 3, 4, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_768c972b03a389502e72873a3e86a2fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 320, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_43e558954f8dd84d2ef1e1d2702f8705(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_768c972b03a389502e72873a3e86a2fc
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ccf813327c1df683ffd36b417520bc5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 320, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_457280032f7128c4b2520b4400c69a4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccf813327c1df683ffd36b417520bc5d
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c5c68140544f0c07cd00234ed62376ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 1764], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_441c510a5878cce176d02567fd24fd30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5c68140544f0c07cd00234ed62376ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1764], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7cd2c5c365f1bc639b05474cb61c5ddf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 1764], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d7d3d40597dfb32ead5b292cc5961b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cd2c5c365f1bc639b05474cb61c5ddf
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1764], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_007f0eb0cc5b8014b740db1d961f6253(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 5776], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e0773b4d8bcaa7a7c21cc87633823c4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_007f0eb0cc5b8014b740db1d961f6253
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 5776], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c00cd60caf07dddd01fb3df6bd2469b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 5776], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bdd4c2e8cf17a3e86e8a6f206df95d70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c00cd60caf07dddd01fb3df6bd2469b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 5776], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0127ddc84ca5f21b4b5cb37e04299df9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 136, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f2ce084e15497dcc813ceb5d59744818(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0127ddc84ca5f21b4b5cb37e04299df9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 136, 160], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f7ef05586058d49cce8b09628e3017f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 68, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_068ed5cd94fbf73f7064f24225da5273(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7ef05586058d49cce8b09628e3017f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 68, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a4d7e8f3b02c4908d9f8552547fc80b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 34, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe9157fd572b843dbdc4c4756e1deb5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4d7e8f3b02c4908d9f8552547fc80b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 34, 40], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0c067404a129af1d25fef40ec7a90b1d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 17, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a2a30ce6246e2d8a5953022e05b1ee7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c067404a129af1d25fef40ec7a90b1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 17, 20], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b27b308c03992b320510bc029faafa42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 9, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ba83014973824bb72ecf4bec51bf5e72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b27b308c03992b320510bc029faafa42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 9, 10], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_80d8ab86619c1daa187f74da05c003ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 136, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53521a1e4d680b0626c8352dd82139c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80d8ab86619c1daa187f74da05c003ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 136, 160], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a80e54a615d54581e1efda91c569b40e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 68, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c8550b3bcf69e8fc904e0645789edf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a80e54a615d54581e1efda91c569b40e
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 68, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ce1cce585511baa0bafcc950a47abdd9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 34, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cbca05003ea4e2b248a3f587b8ddc303(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce1cce585511baa0bafcc950a47abdd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 34, 40], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ba90740b6b704d20ba50a6b6618e91e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 17, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f43c7827d2545c407951e095c7eda2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba90740b6b704d20ba50a6b6618e91e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 17, 20], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_af2b84fd51183f8c28f7b18aece0364f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 9, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_289422d28418cf1bbe0742b7dd50dace(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af2b84fd51183f8c28f7b18aece0364f
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 9, 10], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1b6c8293f65831c9334a0fc0cb0ee6f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 1296], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_628fc1570ce06ad36bc26bb5aa83f337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b6c8293f65831c9334a0fc0cb0ee6f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1296], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6484eab9bbe32febc07cad4cae37bb83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 1296], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_abeff39a31f853d68d6c0cebf85854ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6484eab9bbe32febc07cad4cae37bb83
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1296], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e3ffebd65568785065b6f509cc63465f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 1296], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_37fed06cfd4ed91c2d39a68112a61635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3ffebd65568785065b6f509cc63465f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1296], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5873366485593aa1e6a050d8ea9217b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 49, 8, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_da9459eaa5a2a6a98f89c0aa0a2fdabf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5873366485593aa1e6a050d8ea9217b3
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_da9459eaa5a2a6a98f89c0aa0a2fdabf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5873366485593aa1e6a050d8ea9217b3
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1bb08b3d24e669aaf94be7af5543c94b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 49, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d3fe51337709afe6757595ae4cd2381(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bb08b3d24e669aaf94be7af5543c94b
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8349f2a49b9231f7c4e61632f7b22ed4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 8, 49, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c49e4032eb9570737c6e01612843de81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8349f2a49b9231f7c4e61632f7b22ed4
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 49, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_37a94d9e707cd865f217181b17391e87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf731e6d53dd28615d9afec54bc3bf57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37a94d9e707cd865f217181b17391e87
    def get_inputs(self):
        return [
            paddle.uniform([8, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0877b5531e32f33a383f53bd157ac37b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2401, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c58e8e18a4f96a461ce833a84137c75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0877b5531e32f33a383f53bd157ac37b
    def get_inputs(self):
        return [
            paddle.uniform([2401, 8], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c50cd402ccb228b05ad43e6e5e81bd3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 12, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ab879dad588ea6fc42f4a9e48b91a76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c50cd402ccb228b05ad43e6e5e81bd3c
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 12, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7ab879dad588ea6fc42f4a9e48b91a76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c50cd402ccb228b05ad43e6e5e81bd3c
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 12, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9c697a05358bea34b58d24b470a85fda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a45969eed211697c76986febd50a836d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c697a05358bea34b58d24b470a85fda
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d8e9f28a4ce34fd76baf94bb2d1d324c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 12, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b076779f9a71e1f720cc6c7ba7f1e850(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8e9f28a4ce34fd76baf94bb2d1d324c
    def get_inputs(self):
        return [
            paddle.uniform([22, 12, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6ecfc7e48c9f90e0850385bce9fcaf98(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[12, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c027c630614cd4c692073f563ad7a47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ecfc7e48c9f90e0850385bce9fcaf98
    def get_inputs(self):
        return [
            paddle.uniform([12, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_db036769e2ad1e7d56fd5962db5037b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d76f8bf7d65bcabb26fda085e27fe2bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db036769e2ad1e7d56fd5962db5037b3
    def get_inputs(self):
        return [
            paddle.uniform([256, 12], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8512bd3114b7230ac053059ab21a7909(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 168, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d199e3902bb9eff3451c597d6f647f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8512bd3114b7230ac053059ab21a7909
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 168, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_13efba799831ba1aa390311a4f08c4fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 84, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9ba7b67655ff42abb574a56286e9f8c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13efba799831ba1aa390311a4f08c4fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6e3c1d70017703f35d3bbbb2396cac84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 42, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3d63a7285dcb9535ed9bf9c24a4785fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e3c1d70017703f35d3bbbb2396cac84
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cb275b34ab43129a93a2f7c5477f078a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 21, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_56141c571415542ddfc24b21ad99d21c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb275b34ab43129a93a2f7c5477f078a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4f8f98bda767ec9a08d1da019e7a1692(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 11, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c08be8910a419b9f2f4409ec90e95152(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f8f98bda767ec9a08d1da019e7a1692
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_66324211933c13de12f9da59eae6762d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 168, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_409d08e649ae5da761e9c9cb1f4fe17c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66324211933c13de12f9da59eae6762d
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 168, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c1cd6e6deff5cf978bbe15ef302ff410(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 84, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_593be1eff72fef57024c8f00b55ca3ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1cd6e6deff5cf978bbe15ef302ff410
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 84, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2afa636170cc821e253a1fb2761f2c61(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 42, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b5e2a3a8f624534dab623305fe6eafa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2afa636170cc821e253a1fb2761f2c61
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 42, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8e489f1e99d59a61126b341ac3ae44df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 21, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_677254caf92824a623ec75da31e1fd1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e489f1e99d59a61126b341ac3ae44df
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 21, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a1144ac8a6473d933300e925b1b922cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 11, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1d6e5d4bbcb36776ad1c7e9bfabce1ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1144ac8a6473d933300e925b1b922cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 11, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_80be6c6f32872568cb8aafb48c951313(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, 65536], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_88aacf3cd3f7b50fc69b0860fed9b2a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80be6c6f32872568cb8aafb48c951313
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 65536], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8b596c1d8d87e902dcbe04cfad547765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76dee8755d5dd20cf8c34a12ce9eb7c2
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5bc0e9cc69a43f17541e794c2106bbf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e1bfc6372783f01f98ed165cbe5693c
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 784], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ea094c91674020391a25acb9352d7b6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9fe8580233d94cf2306a30c8060653e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea094c91674020391a25acb9352d7b6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f31cc463b3c7cfaf662022ee75374305(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 2, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d821b496a46e16335f1f7672ca400764(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f31cc463b3c7cfaf662022ee75374305
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 8, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7403db6d7a89fa0d9a7d07bf52e5d599(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa5e0ef54521746d88c5c665ee27d07c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7403db6d7a89fa0d9a7d07bf52e5d599
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bff2ac97d1f50cd3c371e50f9a590d0d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc0f28cb22d614b141c071282680ab73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bff2ac97d1f50cd3c371e50f9a590d0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_91837bfe3f4c5b6ca4739092fcd3eb11(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[576, 2, 96, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e0d3d1091a0823f397eeebd5f3f78b06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91837bfe3f4c5b6ca4739092fcd3eb11
    def get_inputs(self):
        return [
            paddle.uniform([576, 2, 96, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_27a475bfa30153be56fa2378ea51d80e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 96, 1, 1, 96, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74f8853d20ad4cd6639ec117397aa092(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27a475bfa30153be56fa2378ea51d80e
    def get_inputs(self):
        return [
            paddle.uniform([6, 96, 1, 1, 96, 48], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_48a4ff4883b2b1c63f9403eb1f027eba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 176, 264], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2cfa77b7a94421e61c621c27aab602a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48a4ff4883b2b1c63f9403eb1f027eba
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 176, 264], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_db10f8c5ecad126da2b59f0af899623c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 88, 132], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_25c5135f0b2f34fe09dc543ec05df385(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db10f8c5ecad126da2b59f0af899623c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 88, 132], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_915e098e1d32d8d935f3b03b0f1535c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 44, 66], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4caf10cad43c63506b26c452f9ce74ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_915e098e1d32d8d935f3b03b0f1535c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 66], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6ee36dff715dbbb15f0943d5f1e0b8ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 22, 33], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_44a9116afac6ead8e845d6b956da4ad8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee36dff715dbbb15f0943d5f1e0b8ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 33], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6c209b1a077e3b3c70384450474a502a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 11, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f56ef0ca4ff31ef0df967b411811fcef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c209b1a077e3b3c70384450474a502a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b11306caa279b647442d9aa76c4636b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 176, 264], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c10a9ffc006ecb074ef5feb95c58367(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b11306caa279b647442d9aa76c4636b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 176, 264], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fd86645fd2e06be3a4690944602ce701(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 88, 132], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_239844d64203e24ecadf64a5c9925441(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd86645fd2e06be3a4690944602ce701
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 88, 132], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_89a655b83039e313be97e4657a80d300(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 44, 66], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_51f75a4fd03c15af2e4ece955d6dd1dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89a655b83039e313be97e4657a80d300
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 44, 66], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_75508c87eafbc3169991cd511bf194a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 22, 33], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc7d18a24bdcbeccbee8449d50e5abff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75508c87eafbc3169991cd511bf194a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 22, 33], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_268ec478354039d6eb264fc6caca2726(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 11, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a03aa07092dc56fb7921db4a8aaa3c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_268ec478354039d6eb264fc6caca2726
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 11, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4e8d9b63e2c91b585c5fed809877a683(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 20, 128, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f98830438f983e45bda5183763ee1c19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8d9b63e2c91b585c5fed809877a683
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 20, 128, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_892f8c36460cc78452c2435bef822a26(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 40, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf4eaee4be9a6a8de201abe8012687e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_892f8c36460cc78452c2435bef822a26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 40, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_23b4cd10e1ef8ee982a07ee0d5b70e73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 80, 32, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b53726d902eb5bccb28a7e56120639d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23b4cd10e1ef8ee982a07ee0d5b70e73
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 80, 32, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_06ffe708a0cca10cc183bf23d601a5e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[528, 4, 96, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a9a7e58f278842b16f5090ff4f97929a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06ffe708a0cca10cc183bf23d601a5e0
    def get_inputs(self):
        return [
            paddle.uniform([528, 4, 96, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_62fb83302cfa4bef7f18171cd80aa468(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1, 24, 48, 2, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a27c81e6fe807fd011cab5f13e2219b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62fb83302cfa4bef7f18171cd80aa468
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 24, 48, 2, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f8285ec42634319f2e049a5aef4128fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6dcf6262fd8877df9edefe3617aab43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8285ec42634319f2e049a5aef4128fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e4c8c29c2d3a3ab4dfde3d866a9a652f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 116, 32, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55d59560287106fe2ec5614a5f3e937e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4c8c29c2d3a3ab4dfde3d866a9a652f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 116, 32, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0aff650352063f433d7f9b68fdca4414(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 21760], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_daf604da968e956e03e41a4bcd4e2a11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aff650352063f433d7f9b68fdca4414
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 21760], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_70aabea4fe49d7d2c3efb61aa3a1b6d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21760, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09b33658911f7893d4093166f36fc2d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70aabea4fe49d7d2c3efb61aa3a1b6d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_daf604da968e956e03e41a4bcd4e2a11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aff650352063f433d7f9b68fdca4414
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 21760], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e62cede13c47b0fd075f7db28cfa007a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 196, 8, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_79d7b334145f4dfd08a7c3932c396aa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62cede13c47b0fd075f7db28cfa007a
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 8, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_acceec7afb49cdaace12f80dc01a26fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 196, 8, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d073e0eece0fd34ab0a0866ef43aae79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acceec7afb49cdaace12f80dc01a26fc
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 8, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_76673f5e72e3668402cdedc421ee1849(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29d61a42275d6a4289b8be3acc810c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3145fc8dcf54cc47a4c580d50f364469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06f827971a8f84f612d65cc80006e776
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_60def92137baf16976c789140647063e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0617dcf33c4fe3b1979074e17e5dbbd
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8f6cc4f8538fe217a0a20e32b2614dcc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 324], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d975f8a33755598131cd02dd96fbfdfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f6cc4f8538fe217a0a20e32b2614dcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 324], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fcff9d7ee34b55bd04d3aeb43a9e4fdb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 324], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49352f059314b9df0be7a61671a6425e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fcff9d7ee34b55bd04d3aeb43a9e4fdb
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 324], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8a9d0e2c307efff9f565707ba5afb8bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 324], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_42459388f0f9767c9ee6cf0e3d27bde0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a9d0e2c307efff9f565707ba5afb8bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 324], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2b26024e6f896588df7a7f7bd5127c0a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 198, 3, 3, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_607086058495c816b782e11a1e5345cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b26024e6f896588df7a7f7bd5127c0a
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 3, 3, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d1f27af9693ebf06f61ba32fb9cc3053(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 198, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_466df20bd92eac9cfd766ca5048cfbb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f27af9693ebf06f61ba32fb9cc3053
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_92c72fa054e382305af6e24ebc5d08bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 198, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7f0bfced8d6769651ef88ff2bd209a16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92c72fa054e382305af6e24ebc5d08bc
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4f4acf3d5f02491adbefa653fd95f255(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23ba1039ccd174623b9c6bb72b91a50a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f4acf3d5f02491adbefa653fd95f255
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d4a326d2f9474a5feb5689a8aaacaf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e619824f41249824280889dd4e6db702
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c5c3865ed6f5d8d4274bfcb2cd2d2d5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 8, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bd5229125656c1544d8827edd73673c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5c3865ed6f5d8d4274bfcb2cd2d2d5c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 8, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a7c24eb603b2bf25499d4320613bf8f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 2, 8, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80a14e16d3b912f1addb73dd1f435b18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7c24eb603b2bf25499d4320613bf8f2
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 8, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c7460546426e275723552fc9e013621e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a6bb48e1d7b55df527620966ce6addbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7460546426e275723552fc9e013621e
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6f2bba2a672ed20c03ace2044e9549a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_81bb440f1d9e6d898b082c97d39fe699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f2bba2a672ed20c03ace2044e9549a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1e2570e41060c51200cb421154d7b92e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 15, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_01f80a8d6cf2c9e6515151003c4e7c19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e2570e41060c51200cb421154d7b92e
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_80ec024025df99aef42716943aed939c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c6ed4e3a21a5669fb1560b5dde0f9a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ec024025df99aef42716943aed939c
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03782e4d944f1e61a5e270d8dad34919(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d801a3f63884f9364bf925bb56c70c27
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5e3e8f1d92a72bcdcf50c2fd7f82d9ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c0688ceb8a47d57e27f99c90a19bb9b
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6f9dc8d739d8704d8b4a910ccfe9cfd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dff5785c503eefcb9c9b84253e71133
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03a63373c6e78cbf2b3e106039da6c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f8042f2a9c21448974db77e26270663
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_01bf2b73fe61c9592942475a783ed138(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12174faa73e86c03464ec8a20a68da9e
    def get_inputs(self):
        return [
            paddle.uniform([43, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_691aa767a7b2c596e448604ce5f0e91d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 200, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c8e2790b7ae0ee8b3c1cb26cb076cc4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_691aa767a7b2c596e448604ce5f0e91d
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_42f5361d1920d72c3ac9124b0c96d727(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 128, 100], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6fb8a5d9764f69bf7ea2fadba48ca316(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42f5361d1920d72c3ac9124b0c96d727
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 100], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7e6b0473d74e0699d5e2d24152598a65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 5184], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f02c19977ecc8ec6082d943a1f1fc84a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e6b0473d74e0699d5e2d24152598a65
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 5184], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_48550eb47e241be7e5b76587382f5255(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 5184], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3c2b4127f9bfd5aad75c888e29d4ae0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48550eb47e241be7e5b76587382f5255
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5184], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3d43840fb8e9931b68edbf8df54b777e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 5184], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a6083bca6e1b2fd54e47869d0367c97e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d43840fb8e9931b68edbf8df54b777e
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 5184], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_80363010a1f116d3982d083987d5a0e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 196], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f19f6e4979b8fb2e79a788f4522e74a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80363010a1f116d3982d083987d5a0e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 196], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cf415c498cbed0506f631419fd01ece7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31adf99fcd3dc5d1ee65c307e60599d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 196], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_272725b753856e4a5137f597237b3d67(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ecd01093163b1e685f551d6ac2151fdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_272725b753856e4a5137f597237b3d67
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0c99416cbf0cabc11d5a9f6b62091af8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 32768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02b8304772b8e5c8a888b98af9024f1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c99416cbf0cabc11d5a9f6b62091af8
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 32768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_90ad315cd08e56baec1c7e0b247f5a9f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 19, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60f9f9bc94f917cfd13b054edb30849a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90ad315cd08e56baec1c7e0b247f5a9f
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a7d1b0268d349ee63c28b2a6f5f09263(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9874f711042e2bfd107011a52d643e65
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_91ce96f4f675c6b3875a1bad1955a935(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_736429e2d6de83812117a67bd8bf0eec
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4884a40b93594d469e258132bdbdd56f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c9267bec5ee55f9afae9c6d978b6d09
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d2bb59b7f4a71d24aa04e65dafd9223c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a254ccf070edb72f7535e571a52a7ae5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2bb59b7f4a71d24aa04e65dafd9223c
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a258dd145531ea97190a8cca89565af3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 3136, 3, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bfb09ad05cd9291a1ed2f49db6f39ee7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a258dd145531ea97190a8cca89565af3
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cf4a2b21b5d3cf8f17e37d73fa57ab96(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 3136, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4b437df0712f43d917946c5a9673a1ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf4a2b21b5d3cf8f17e37d73fa57ab96
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_031d36a3a5a0ece5a88d8d20d0a5c90c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 96, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_becfa8f6dbf5b84b87cce88294fcb236(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_031d36a3a5a0ece5a88d8d20d0a5c90c
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8a1994d3cc6f890cb935370af05a2ce0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 2, 3, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bb36b0721490a8053e09bc9ecc78f230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a1994d3cc6f890cb935370af05a2ce0
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b1620b559ab272f1355a37d28ad3736f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 3, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bd6970c60989a1dea53ea2158a4b6683(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1620b559ab272f1355a37d28ad3736f
    def get_inputs(self):
        return [
            paddle.uniform([43, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bb7f956137ad38cc22acf3dc633d470a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 8, 7, 8, 7, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afcc65ea701f91a6344b01d1ad0664d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb7f956137ad38cc22acf3dc633d470a
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 7, 8, 7, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_76cefa5e02fef6152f89c889e46792ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 64, 49, 3, 3, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_94b3aa6aa6355ab26e26c83ad6cd4076(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76cefa5e02fef6152f89c889e46792ee
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 49, 3, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_68bdbe2c601197f5a6528c3b85cf0f2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 64, 3, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc53022d003d982f9250895d1d60a92b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68bdbe2c601197f5a6528c3b85cf0f2b
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_da9459eaa5a2a6a98f89c0aa0a2fdabf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5873366485593aa1e6a050d8ea9217b3
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0279ba42dd029a85640b4f3c3b71e93c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 196], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d0613dfd951f12a4a7a0c7b0abff0d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0279ba42dd029a85640b4f3c3b71e93c
    def get_inputs(self):
        return [
            paddle.uniform([8, 196], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2f60a274c4db64220fd4a2093b0f04c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[9604, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_45e5e764ad6c599ee6920b2e877a9f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f60a274c4db64220fd4a2093b0f04c2
    def get_inputs(self):
        return [
            paddle.uniform([9604, 8], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_71acc1dd4e0b1287d85016dbfd1dd6b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 8, 196, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7f379345b632d45db6c5b0beff1c36a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71acc1dd4e0b1287d85016dbfd1dd6b5
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 196, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_70e8e4ab314e9a804533376c74611d5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 100, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c442d60752d20d4172b6d201e48e67f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70e8e4ab314e9a804533376c74611d5b
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cccc826efb628b41f87775e3944f9f13(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 256, 50], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_96cc0c9f355fe9d5fd0f888a1ebc8fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cccc826efb628b41f87775e3944f9f13
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 50], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a112a2636ecca3ca355dff12945f8dd6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[240, 4, 96, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b1ba62b83b8b624b7e3180f42029a59a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a112a2636ecca3ca355dff12945f8dd6
    def get_inputs(self):
        return [
            paddle.uniform([240, 4, 96, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_66c6c6b526ade27b1a77f4afbe02a176(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1, 24, 48, 2, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_992640f55e91bf7c460fd636dfb34571(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66c6c6b526ade27b1a77f4afbe02a176
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 24, 48, 2, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_051734f99c6c33b33858c391dfbcc053(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_343f7d8725b39b02a9cffe0ced47c44c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 676], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_870832a35002064bd31e1389c6b90bd9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 76, 676], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e9ef19fde993c5ff3a52a62f48e222b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_870832a35002064bd31e1389c6b90bd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 676], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_385fd542618f80dcbfc9567baaca9e40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 24, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f80d70bf5d4fbb5c1e419ffd8c79350d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_385fd542618f80dcbfc9567baaca9e40
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_867c60590970bc95bb0e4fb82b534813(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 2, 24, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1d18e4d693cf56500edc85c4cbae66a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_867c60590970bc95bb0e4fb82b534813
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_79155854e8f8ff91e499fb119522bbdf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 24, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_581160e7a0244e40c03be34e1312b7eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79155854e8f8ff91e499fb119522bbdf
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_92bde32ef0726228edc481e9b579de67(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[12, 8, 288, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fdce69c973b8e9584563c04bb619a1fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92bde32ef0726228edc481e9b579de67
    def get_inputs(self):
        return [
            paddle.uniform([12, 8, 288, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cc34476197396feb05f58aba30a460b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a18a2355f862818d6ad8b6fe7b9636dc
    def get_inputs(self):
        return [
            paddle.uniform([6, 2, 1, 12, 24, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f70a1bb3d43b1e8da4056892e6a221e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 100], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e5fba102d05141d64dac3282e3ed6b5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f70a1bb3d43b1e8da4056892e6a221e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 100], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_57dc0da21db17ac38030326ddd715dfc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 100], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e26802210220685d1290eb67daa864e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57dc0da21db17ac38030326ddd715dfc
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 100], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4e67ca82bb4d9ab8bdb728d4bfb068c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 197, 2, 6, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e96e5744f2d61305591f127e127b5dd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e67ca82bb4d9ab8bdb728d4bfb068c7
    def get_inputs(self):
        return [
            paddle.uniform([22, 197, 2, 6, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2a8afe6affb6234c3f0117b21a0a33b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 197, 6, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d5b1a8812fa5c3a44b612f5e3d8b5f9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a8afe6affb6234c3f0117b21a0a33b5
    def get_inputs(self):
        return [
            paddle.uniform([22, 197, 6, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f35a4b9eaa6098453feb4f89ed9723f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 6, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40fe8b3bcc40e672b5f5a5e62fc1842d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f35a4b9eaa6098453feb4f89ed9723f7
    def get_inputs(self):
        return [
            paddle.uniform([22, 6, 197, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0a44f2fce8f2b57e973c2ddf5c791ffc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 200, 304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_262c877cefbffd7e5b74b410da133e78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a44f2fce8f2b57e973c2ddf5c791ffc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 200, 304], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a8119ba02cc79389f478c9b76e99dc7e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e2aa1eaa8cbef55f832fde108b6ca54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8119ba02cc79389f478c9b76e99dc7e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 100, 152], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7c56a7ddc3dcef04109fae33a4d070a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0609ad25f79da6c30b05f840ece7c1dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c56a7ddc3dcef04109fae33a4d070a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 50, 76], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ad7bd895e858e11d0d8ec59d02ea971b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e398cdf8fa8b559717b7ce0a2bd73ec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad7bd895e858e11d0d8ec59d02ea971b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 25, 38], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_20aef1c1c84cbec37e6bbd57c401c94a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc51b905730371c3ed2389636dd2c82e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20aef1c1c84cbec37e6bbd57c401c94a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 13, 19], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d149104814d252c747d4356628171ff6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 200, 304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ed44d88f7716e25824418a5765c6643(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d149104814d252c747d4356628171ff6
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 200, 304], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5ff3338aab435cc3f867c9ef65aa085d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d98ca315d512703222d74f47c58f278(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ff3338aab435cc3f867c9ef65aa085d
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 100, 152], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c5bd9bc1d2804a7f9e93b749a3532a94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1f838a40c84ac67e248e813f0cc459c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5bd9bc1d2804a7f9e93b749a3532a94
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 50, 76], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a4b3fd43645a9cce8b6ed1de35028392(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3069b6feeb55003a02526f6a9c10e7fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4b3fd43645a9cce8b6ed1de35028392
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 25, 38], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4f2f15389b0f1b5b9ccf49161f4b5175(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1ff828c1c72a414c906d1bc6ac9bca9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f2f15389b0f1b5b9ccf49161f4b5175
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 13, 19], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2cfa77b7a94421e61c621c27aab602a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48a4ff4883b2b1c63f9403eb1f027eba
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 176, 264], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_25c5135f0b2f34fe09dc543ec05df385(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db10f8c5ecad126da2b59f0af899623c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 88, 132], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4caf10cad43c63506b26c452f9ce74ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_915e098e1d32d8d935f3b03b0f1535c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 66], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_44a9116afac6ead8e845d6b956da4ad8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee36dff715dbbb15f0943d5f1e0b8ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 33], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c08be8910a419b9f2f4409ec90e95152(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f8f98bda767ec9a08d1da019e7a1692
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5c10a9ffc006ecb074ef5feb95c58367(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b11306caa279b647442d9aa76c4636b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 176, 264], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_239844d64203e24ecadf64a5c9925441(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd86645fd2e06be3a4690944602ce701
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 88, 132], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_51f75a4fd03c15af2e4ece955d6dd1dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89a655b83039e313be97e4657a80d300
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 44, 66], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bc7d18a24bdcbeccbee8449d50e5abff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75508c87eafbc3169991cd511bf194a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 22, 33], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1d6e5d4bbcb36776ad1c7e9bfabce1ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1144ac8a6473d933300e925b1b922cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 11, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_049b618a25a7043d0af5276b85d4902d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_82c521ae50826bd76ee386f723c41707(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_049b618a25a7043d0af5276b85d4902d
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e3206f7185afb1dc5e0a64266fb51b51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e6a2aaa7eef3be52cf716a2cc4a17e3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3206f7185afb1dc5e0a64266fb51b51
    def get_inputs(self):
        return [
            paddle.uniform([16, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7ba9da62a85a06a1c0f2e65f20810e92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[784, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23ae2857fec62834e692d6911e060692(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ba9da62a85a06a1c0f2e65f20810e92
    def get_inputs(self):
        return [
            paddle.uniform([784, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d6a304c8d2dc4ffe90823b3921654dd0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 49, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_10effd1c7af701152a8f8b531f8c10e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6a304c8d2dc4ffe90823b3921654dd0
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 49, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1264d0e662ce353c9eb4552bdf62c982(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c1527eec68163910f412fc52da11d63
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1600], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d49d2a75214fe20168073e6e4d4008b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 1600], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d00222a7415abbf67632a3e5e300535e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d49d2a75214fe20168073e6e4d4008b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1600], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_448bd9ec0f6b78585c8656363a91f603(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 1600], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a57f566fed511f1a06a1489ba1c60ce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_448bd9ec0f6b78585c8656363a91f603
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1600], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8f383dbc13ade5f8c112a5e34b8eeecd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 144], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8bdd43ae3e192fd763f1b363d58de1b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f383dbc13ade5f8c112a5e34b8eeecd
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 144], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a0aeba2c31530c1bf04644f0cb1fbfef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 144], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fca25286a6a152c773a0b37dd14acb3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0aeba2c31530c1bf04644f0cb1fbfef
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 144], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_07c30113b3c1cdb0fdb34058a45dba4f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 36, 28, 50], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e9e47aabea1012033474061b9a833d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07c30113b3c1cdb0fdb34058a45dba4f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 36, 28, 50], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_24fa89f9e243f08ae2c8f4a3926f4c00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb9f70282899345810179c07997ffef4
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0c2deb8e07905eb4b1bb495ded91337e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 192, 784], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_acacfe9364264aad11d937034363378d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c2deb8e07905eb4b1bb495ded91337e
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 784], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_db168f22dc31d798cdc32e7ceee1c9c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 3, 12, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2106d325f9b82b91c465f2f56ef5a9fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db168f22dc31d798cdc32e7ceee1c9c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 3, 12, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_01aaa29f674ebfe6cd972d52bc4e034e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1174, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_241aa34d878817a0dd5d1f57a0d2edba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01aaa29f674ebfe6cd972d52bc4e034e
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_32af6bd3107bb7be8474801d88c7ec81(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1174, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fca86891606939092a5e0ab38c4e35ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32af6bd3107bb7be8474801d88c7ec81
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6c0289ebf5e4ed59e194ad32b4dc2244(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8cd3a659fcdfab9ad136ee4702248603(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c0289ebf5e4ed59e194ad32b4dc2244
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88c8fdbde53013d6879a05b32e43f9a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82aa51e07a4573708aef60fcd33c2833
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 289], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d778c7e6041fc4f84759767a43fb225e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 289], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_679c5060f6e05470e2e8d62674d719f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d778c7e6041fc4f84759767a43fb225e
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 289], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_47e9518fa8d3423f1a4eb32c1d5076ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 289], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9a7498bccc1b0945134fbb9ad43e77f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47e9518fa8d3423f1a4eb32c1d5076ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 289], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_566c2904724b532696c4caefaf816cd5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 1600], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_178b1f28ec00b7e5e7fd57094f448ef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_566c2904724b532696c4caefaf816cd5
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 1600], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_62fd29f80e8d31a187a735939a6d8016(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc4e16bdca22ccda1ee9297bb7762fd5
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1600], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c1320b33acaa23de10bfcbfb7345f998(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 32, 144, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d462bd8193390c5cf65f237c6c92fc96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1320b33acaa23de10bfcbfb7345f998
    def get_inputs(self):
        return [
            paddle.uniform([6, 32, 144, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_78745594ce305264858c3f517cf6f24e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1, 1, 12, 12, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4544c4dd62d731af87153533e7d2f046(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78745594ce305264858c3f517cf6f24e
    def get_inputs(self):
        return [
            paddle.uniform([6, 1, 1, 12, 12, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9a51c504386f459556682c37e6409aef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ff6cfef19eb967f6b8ae54206ea493b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a51c504386f459556682c37e6409aef
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7c1e1416d893818f1de03187f7eedcd6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 198, 3, 3, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6e10ddb0c8acf88de63808577fbd01ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c1e1416d893818f1de03187f7eedcd6
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 3, 3, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cb568aa5b4189bedd9b0f56484dd1b3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 198, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c350952e89109ac367346ca5a299bff2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb568aa5b4189bedd9b0f56484dd1b3c
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_973be8ca93bb921497b74593c93cb1d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 198, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_458dcc1eda1e36e309c1d704916aff5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_973be8ca93bb921497b74593c93cb1d1
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0146e318a3ec2a9a2a35abef5c497578(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 32, 144, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c606111251ab76a875eee1c1f38bdd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0146e318a3ec2a9a2a35abef5c497578
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 144, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c35646adc6a905cabf4ac542a1f4d40d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1, 1, 12, 12, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_78f483c228fb6afbd599c99cf97ef2f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c35646adc6a905cabf4ac542a1f4d40d
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 1, 12, 12, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f98830438f983e45bda5183763ee1c19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8d9b63e2c91b585c5fed809877a683
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 20, 128, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bf4eaee4be9a6a8de201abe8012687e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_892f8c36460cc78452c2435bef822a26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 40, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7b53726d902eb5bccb28a7e56120639d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23b4cd10e1ef8ee982a07ee0d5b70e73
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 80, 32, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7b4b5f2b67b5b35904c6990124782d74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 160, 16, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c0caa810d29d98354087580bfefcb47d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b4b5f2b67b5b35904c6990124782d74
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 160, 16, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_19267aef56f678aed366fb18810c7bf5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e7af29ebd5d129c2a5a002433659e072(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19267aef56f678aed366fb18810c7bf5
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_231e35d7454c3cf4c2eb931569b05d13(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_33750b9e919c90e423f5cbcf1d3929f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_231e35d7454c3cf4c2eb931569b05d13
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bfa4a73154cc9fd661fde3f846f97d67(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_01ca69f7e91e543cb2ec7623e061c00a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfa4a73154cc9fd661fde3f846f97d67
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d8d05272f39c9467c5f3664f33ed070e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 441], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_47bcff0601946bd75b898e6bdf62d482(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8d05272f39c9467c5f3664f33ed070e
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 441], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d65f6f30ff0da12ab56538d3eb16821c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 441], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6d248a19c0171d6845d51200505f2be6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d65f6f30ff0da12ab56538d3eb16821c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 441], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1430289c7531fc62ba5efa5888b7e0d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 8192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fcbe5bd95709eae55713f5e9038772b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1430289c7531fc62ba5efa5888b7e0d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 8192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4adb6a64a0ae43ada89686a8a2304d9f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 8192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ce7dcfcab781e3e3c91f7c606285099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4adb6a64a0ae43ada89686a8a2304d9f
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 8192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c5876ad700e980ad6c027aa3a1d515a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c69235fd3d31c63159f07f5ad042bf71
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_14069fa497fe6b86febb04e5ae40660e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fb31d094233c7b13fc7ef772520913e
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_281cd022c91dbc648010c8d5daebcbd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c85fcaddcc667107ada757cc01b54fae
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_769ebe0f31eee0529acfc9122ef40172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ded1a7c05af155087ddb75b3ab878e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4aada27a760f71f394e5e9c393d740c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_861b55d23659bfb7e30bb07acda5fed3
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_55622116095b67cd7a8605c3ea883674(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1c5ea8fe5310ea4b3cb7a0cda27ec5d
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_660bcef112fbf7db74c11e7373b8a1d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db74162f4e6a34eb35457836172edf90
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_29245447b95c1da982dd9b88e75b7253(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21259e70f761ede775e12e07f6f854
    def get_inputs(self):
        return [
            paddle.uniform([11, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e4542efefe8f666234f64f3301bc1ef7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 1156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de0af9c5e7d2ace23e487cbae339baa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4542efefe8f666234f64f3301bc1ef7
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1156], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c7a2ba69166fa9116f9176e8fdebfb74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 1156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ea6e97b682e7cb50d2188fd32d5b599(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7a2ba69166fa9116f9176e8fdebfb74
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1156], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e6bc6d6b53ad9041b8996539f44e9f0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 1156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_77e4f96bf7fcaa86f303b54926b95b1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6bc6d6b53ad9041b8996539f44e9f0b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1156], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8fd0403ce8760c92cb79a69b21de8481(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 1444], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_937c2afd14c2a799915d9b9ea610b4c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fd0403ce8760c92cb79a69b21de8481
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1444], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ec83816c96512b549bbd39ee7ff5a81a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 1444], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ecb79d4eb5c7179bb5ecdac07ac6087e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec83816c96512b549bbd39ee7ff5a81a
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1444], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_188efac39328cf8b7fceba4e5dc2464b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 50, 3, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fcad1848023735016cb7559dfa0984bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_188efac39328cf8b7fceba4e5dc2464b
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 3, 8, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7ed745f7f86f6dd1c5a7236ed555b0d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 50, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e83e002a23dda76de2ac7eda9ded82d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ed745f7f86f6dd1c5a7236ed555b0d5
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_116276ba383e3639512ce3b72a1c4dc8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 50, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f15e4f43e9e39d85be58f9649dc3aee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_116276ba383e3639512ce3b72a1c4dc8
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a84564e60d025ac9b5b2ab5397fd4962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_581351d2c9f222d4e806d2f588dfd829
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_49d1af5fc0f630b1908640a479a3a8f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2999c53f56f06515655debf406d68a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49d1af5fc0f630b1908640a479a3a8f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d10bbc13eac969e4ab1bf7e029eba0c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_284dcee3f0bbf574c263ae5bf0bc186f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d10bbc13eac969e4ab1bf7e029eba0c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_92df7e6af3f814c7ae5356b8a996b553(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16, 128, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f989c231f879ff5a14dfb8a405abe3cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92df7e6af3f814c7ae5356b8a996b553
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 128, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_76673f5e72e3668402cdedc421ee1849(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29d61a42275d6a4289b8be3acc810c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3145fc8dcf54cc47a4c580d50f364469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06f827971a8f84f612d65cc80006e776
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_60def92137baf16976c789140647063e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0617dcf33c4fe3b1979074e17e5dbbd
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1247e5974bcf831b85bff78035225f53(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 7, 1, 7, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02eee51af78e7e19f807057f2847e0f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1247e5974bcf831b85bff78035225f53
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 7, 1, 7, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_92dedaec40aad3f7c347bb59c85ff802(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 49, 3, 24, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cb63ddee5d3ce032d7d3a8da982321ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92dedaec40aad3f7c347bb59c85ff802
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 49, 3, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f5956615907214bfca4df75c5c979ad5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 24, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9e1726b0cb3bef625078eb72be1e6fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5956615907214bfca4df75c5c979ad5
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6055843f922f66b440362032a2059c55(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 3, 6, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c12014db9a3f3f455dacd0b176182b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6055843f922f66b440362032a2059c55
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 3, 6, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_217d63f7958073d83b51c4fb61ee88af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1025, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d22c95d30e2c68e440407d567de08426(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_217d63f7958073d83b51c4fb61ee88af
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_175a4a01163d8bbe50ef540cf4707016(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1025, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6dc19bdb42164a405a53043e83dc3024(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_175a4a01163d8bbe50ef540cf4707016
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_710e3c50e431fd73026438794cd93038(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc0cc986d9b3e247a75890373830c371(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_710e3c50e431fd73026438794cd93038
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_636e2dde81af971a75ff84073e796eb9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c31cfed443ee32b763fa90ec82030566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_636e2dde81af971a75ff84073e796eb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 12, 12], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_76673f5e72e3668402cdedc421ee1849(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29d61a42275d6a4289b8be3acc810c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6a075d50ad9c4bf912cb6774e8c1b7b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f6c6361dc5654339d27ac533307ae9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 169], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1da26df5301b255b23ca5b8627d7e394(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_398e3e4874eca6a3306119ffa4329d14
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 169], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7834d5bcee60fe411bf1842c4d972fe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1d629783b007af804fbe95a73f06c4c
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 7, 1, 7, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6fd30380c37e45a892ad4738bb059e25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77a7e7996acbc6fcd97f4e0f2af9d257
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 49, 3, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_491ce757e507103a23f7469d0554e25e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ab7b62f825fe7beca19a9aa43990acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_646a62355a4f033c940be394e1e40307(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfd3f30f3ea72d2295750820bd79414d
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 7, 2, 7, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_191b83046528c7e93a8100ee2c26b597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4823e8f104c33c1b0f5551ed26ca94bb
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 49, 3, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4392384ea9afcb19ad621a79cec5a45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0487eb4fba7cdca9eae703b7ccacb830
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e423dc7ea07b04578889d6ea42f7e74e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 784], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4f1a0b9a06b2cc9993e5c5a6216377ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e423dc7ea07b04578889d6ea42f7e74e
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 784], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ec50c8425c07efb66eb45659e5e61cd3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 784], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9ea53b525a5671f10a524920204e182a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec50c8425c07efb66eb45659e5e61cd3
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 784], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8a6a07b9d8c2ae2d9b08dcd058d91dc9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 192, 288], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c4487a38e0a7f1eb658b332e3acb5aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a6a07b9d8c2ae2d9b08dcd058d91dc9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 192, 288], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2c81a6e6581cc06cfe364cffc890013f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 96, 144], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9afc39f4653d75d52f7958fa1b12d65a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c81a6e6581cc06cfe364cffc890013f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 96, 144], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e4e288a939db1e09e043df3c848391a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 48, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55c499598e1dd352d1395db58d1fa662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4e288a939db1e09e043df3c848391a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 72], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_42e71324cdeb85098b19c49740a479b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 24, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2eaf7125c82b96f77fb59b018f9550cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42e71324cdeb85098b19c49740a479b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4e36290d185b2a5cbb0820ca14362a11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b25ecc1272b26e9dd05c968de9744613
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 18], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_88fe47fec67a58d61a4a3d1d16cc3328(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 192, 288], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2cf622367adcea2bec13682f0b78117b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88fe47fec67a58d61a4a3d1d16cc3328
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 192, 288], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3c598154bd658680190eba6eeb28f153(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 96, 144], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59550ffcb855d269f6fd1649a08ae087(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c598154bd658680190eba6eeb28f153
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 96, 144], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_637afb49112ac4682401741d34792b82(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 48, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a4836acaa4f3cbdfc2c824b65dac3fcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_637afb49112ac4682401741d34792b82
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 48, 72], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6bf8b2f21ca5a060d6a86147919864d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 24, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_879f9a4f5406b2b77cc9bb7e3d44aec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bf8b2f21ca5a060d6a86147919864d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 24, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_55f9d105d55eb2fcb17cffdf014619d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b7304a101a8de5f7ec89f91f44c1006
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 12, 18], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d77a2a27951037085a3b255bd345b385(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d29bebeb36d24dd6bee8dcfba7af6f34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d77a2a27951037085a3b255bd345b385
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5c6128d07d00a8e1e438ecafc18d2efa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 8, 288, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_955a093ed4c5cae10a4b24ad73e1c5e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c6128d07d00a8e1e438ecafc18d2efa
    def get_inputs(self):
        return [
            paddle.uniform([8, 8, 288, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9a8d3b5b266194bbbcd4030ef469ecc3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1, 2, 24, 12, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f21166b14ccec604d9e1ed5137f41e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a8d3b5b266194bbbcd4030ef469ecc3
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 2, 24, 12, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bfb09ad05cd9291a1ed2f49db6f39ee7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a258dd145531ea97190a8cca89565af3
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4b437df0712f43d917946c5a9673a1ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf4a2b21b5d3cf8f17e37d73fa57ab96
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_becfa8f6dbf5b84b87cce88294fcb236(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_031d36a3a5a0ece5a88d8d20d0a5c90c
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bb36b0721490a8053e09bc9ecc78f230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a1994d3cc6f890cb935370af05a2ce0
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bd6970c60989a1dea53ea2158a4b6683(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1620b559ab272f1355a37d28ad3736f
    def get_inputs(self):
        return [
            paddle.uniform([43, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_133643707e5a603060a38f0e069607b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_513a7aa4a27595b824e8e952e1bb0c0b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2704], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8c58840bd1568a7a9a6f5eb348647529(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 76, 2704], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a990d756a3db05fc62c6738edc4698c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c58840bd1568a7a9a6f5eb348647529
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 2704], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4b437df0712f43d917946c5a9673a1ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf4a2b21b5d3cf8f17e37d73fa57ab96
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_105258f83ab58884856acffda56f7f0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51f6e4e493661b6052e041f3ff5dbbbc
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 3136], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_55fb65de185b959fb46a620ffe5d5763(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 197, 3, 3, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8a7a31246b13203fbe5871d2245357da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55fb65de185b959fb46a620ffe5d5763
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 3, 3, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2d2b3f418ff49b93d9443573a8df6007(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5148b6fba916c85c922c40006acef53f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d2b3f418ff49b93d9443573a8df6007
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ad9fb1d3139427203d613be91e0d5763(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53c4c78fba8c8146eb961fd5d595c7cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad9fb1d3139427203d613be91e0d5763
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_51b56d5ecdf807a71e270ccee0eaa97b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 7056], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4abe7429488169f130a593b108ace109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51b56d5ecdf807a71e270ccee0eaa97b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 7056], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2a14be703bcfeebda8119ea0c70cdab3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 7056], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a7692dc0f666e83171b1f4d96a06d7af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a14be703bcfeebda8119ea0c70cdab3
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 7056], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_333f54d70101912ab921726d45fb4758(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 80, 144], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d4c6328bcdcc603b58027412e9b644f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_333f54d70101912ab921726d45fb4758
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 80, 144], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_133643707e5a603060a38f0e069607b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_513a7aa4a27595b824e8e952e1bb0c0b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2704], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9813a90a013b07c2d69599597a4e0186(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e25ce0e126b680a08732401d6e34c41d
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2704], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e1b43b443c67d632a75e36f51d7ec694(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 49, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_83cfbf1444e3d80f69c7fa9c8f6fff8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b43b443c67d632a75e36f51d7ec694
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_52ff648e9dd84f5c44548e4bd75fd23d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 49, 16, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b7659501c832e3e7d3a18c4dd045377d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52ff648e9dd84f5c44548e4bd75fd23d
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 16, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_da8e2f1f93670ddd03038ddec1b687ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4bb6e3deb02aeddbeb0c85bd4577160d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da8e2f1f93670ddd03038ddec1b687ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_647e2a486d5af368db11f7223957dfe4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09e1c4dcc071a690efc026e825f69358(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_647e2a486d5af368db11f7223957dfe4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1a30c84ee0173b7456f5b19d882e9111(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b81e24bb92111aa18dc7e4b62ed85c2
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 7, 2, 7, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9d1527194c9f118e732d746c54716b6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58d46d6af14fde8039b960580dddc065
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 49, 3, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4e8944027eefcf98155844dd21093a22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35b94e436c33d86a3020111bc76e2ddd
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d4a326d2f9474a5feb5689a8aaacaf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e619824f41249824280889dd4e6db702
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cb6ca6b66b09ef0568b67b46a7eb4854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fd087e9b8d337f4e1b6470e74a6e682
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9c6b6c0d3a94c2a198ef93a6da959487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a65e69570ac7936e6fe370c04384258
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_34a7921ea0b6032b79b7c1a8af883d41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 8464], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a15f3ed68571c1eb99a5d44bb2b08604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34a7921ea0b6032b79b7c1a8af883d41
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8464], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ab1a86a3f2486242bdd22fff6765221c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 8464], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ad6331abaf2ae5ec94b041a67286f6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab1a86a3f2486242bdd22fff6765221c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 8464], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e08a65ff36de22a726435f401f84d918(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 197, 2, 6, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f51f50292c7f588f408d98c479666cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e08a65ff36de22a726435f401f84d918
    def get_inputs(self):
        return [
            paddle.uniform([10, 197, 2, 6, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_26b4b23e8eb87b507557c1abf9d15315(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 197, 6, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1e75cc1f4fef5393fbca1893f9f52e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26b4b23e8eb87b507557c1abf9d15315
    def get_inputs(self):
        return [
            paddle.uniform([10, 197, 6, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d8df2b47cde8af75fad934f25b03fbb5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 6, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20e7d224edcb416aabcec726037e5670(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8df2b47cde8af75fad934f25b03fbb5
    def get_inputs(self):
        return [
            paddle.uniform([10, 6, 197, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cda9d6cf5a776dc418507f5a12dca5cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 577, 3, 12, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eb6481cdd8f5852efa210a82dbffb37f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cda9d6cf5a776dc418507f5a12dca5cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 3, 12, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_92074a94b7fdbc7fee4e192de5aefa56(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 577, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c1747ddd565a22c427fd019e478d8205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92074a94b7fdbc7fee4e192de5aefa56
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_471ed70a60e333d36570a03c211f3c51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 577, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_169872c9e3bb4bd9983ca5ca5d8a454b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471ed70a60e333d36570a03c211f3c51
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_eea972cf53ab0bb8387ff456c760dc7a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 3136], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7582bf7ccea813792fe30e7c99cdcf6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eea972cf53ab0bb8387ff456c760dc7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 3136], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f8cc7fe59e5958b16e5282f46bd47298(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93fff2ee76cb12b87038ad024d7858eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 3136], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_902b16f142e51167681ae99ebfeb8098(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4be89125fd8bfec5dde5279597f614
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 7, 8, 7, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_16425d7375cb8d5b4c9d3b8f861422fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c09db675753c27b512b2ff6d36a883ef
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 49, 3, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_497c6edf24c50507c6ae0d8d4787994f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a821b35effdac8f38776368a78c6e04e
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d0685b45b469eaad5c219d388ef91aa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93c975b04c598aa3c1e7ff8a4a2939c
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 196], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_89bd00f141b3a0ecd5dc3412d8fc13dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 8, 288, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_46fd6bfe8366c7610738e989718acc1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89bd00f141b3a0ecd5dc3412d8fc13dc
    def get_inputs(self):
        return [
            paddle.uniform([20, 8, 288, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e30c24680d5b1c82f855eb5775810692(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1, 2, 24, 12, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f02cfa28186b52af0f91c45ce8511294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30c24680d5b1c82f855eb5775810692
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 2, 24, 12, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_514ad0e9d1f84538b58ebc86e236420f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 8192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a0c2302d3bff67de6a0060137a70842d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_514ad0e9d1f84538b58ebc86e236420f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d4a326d2f9474a5feb5689a8aaacaf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e619824f41249824280889dd4e6db702
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cb6ca6b66b09ef0568b67b46a7eb4854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fd087e9b8d337f4e1b6470e74a6e682
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9c6b6c0d3a94c2a198ef93a6da959487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a65e69570ac7936e6fe370c04384258
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_da6d9138c3efe0b5c36c28c92c2de891(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 32, 144, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bd4fb1b2698412c3302c0626f52e4d5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da6d9138c3efe0b5c36c28c92c2de891
    def get_inputs(self):
        return [
            paddle.uniform([4, 32, 144, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d70d9245391e82dec3bf9cdaa633bb2d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1, 1, 12, 12, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8a72e81274149ae0a78b1ea8d6363727(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d70d9245391e82dec3bf9cdaa633bb2d
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 1, 12, 12, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d4a326d2f9474a5feb5689a8aaacaf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e619824f41249824280889dd4e6db702
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cb6ca6b66b09ef0568b67b46a7eb4854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fd087e9b8d337f4e1b6470e74a6e682
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9c6b6c0d3a94c2a198ef93a6da959487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a65e69570ac7936e6fe370c04384258
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4735edb7e2f7b773ae387acb8f6d91dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_112b2548816103019986f69677d7b728
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8a8f3cd33c547b0318ba04501b1b5d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291f7e7c973ea004a299570d6cd22ca0
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0e35fbec8aa18f5da5f08a3430d2c6fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 96, 9216], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e351bd141b35eb4039b0d63fea3f7f1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e35fbec8aa18f5da5f08a3430d2c6fd
    def get_inputs(self):
        return [
            paddle.uniform([4, 96, 9216], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_acacfe9364264aad11d937034363378d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c2deb8e07905eb4b1bb495ded91337e
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 784], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f7df3e6c8e619e6ba7633d47e760efef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_635fde2749ed6ffc87b6852ecc1c305a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7df3e6c8e619e6ba7633d47e760efef
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cb6ca6b66b09ef0568b67b46a7eb4854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fd087e9b8d337f4e1b6470e74a6e682
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_90b8be115cc76d7e1abbb94bb858c516(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ce7ec5178d7a9bf4841d03172dd62a4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90b8be115cc76d7e1abbb94bb858c516
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1029a2284d87cfbe7248d1a28bb120ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 320, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bbe2e454df7e5724d35ca8f6ca9829aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1029a2284d87cfbe7248d1a28bb120ef
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cf3a7adb1dcb3f4463b302f62c61489d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 256, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d066f487229e120ad043f6bfa5624552(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf3a7adb1dcb3f4463b302f62c61489d
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 160], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_60ea5d8c4ace25f3e000326b318f7328(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a8fc265653363fb62af2c3858cf7d78c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60ea5d8c4ace25f3e000326b318f7328
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_afcc65ea701f91a6344b01d1ad0664d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb7f956137ad38cc22acf3dc633d470a
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 7, 8, 7, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_94b3aa6aa6355ab26e26c83ad6cd4076(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76cefa5e02fef6152f89c889e46792ee
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 49, 3, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dc53022d003d982f9250895d1d60a92b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68bdbe2c601197f5a6528c3b85cf0f2b
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_76673f5e72e3668402cdedc421ee1849(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29d61a42275d6a4289b8be3acc810c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3145fc8dcf54cc47a4c580d50f364469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06f827971a8f84f612d65cc80006e776
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_60def92137baf16976c789140647063e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0617dcf33c4fe3b1979074e17e5dbbd
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_382842bd160162eddda7d7a178127394(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfb86674f08e2524dd57ca19edcc36a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4624], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c46f09b3c8bd6dcb324d3a4807d46066(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 4624], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_37b1178f48307f07cf74fc94702f89c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c46f09b3c8bd6dcb324d3a4807d46066
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 4624], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e1bdc423af734e1ea8207f8a0f7eb1aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 768, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5347f41a2f1879a82c28a9306eb3b1c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1bdc423af734e1ea8207f8a0f7eb1aa
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_27fa2639720dc55b890ffc91f03b1c74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2420ed4590915e918010772c890c36c
    def get_inputs(self):
        return [
            paddle.uniform([100, 256, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_de0af9c5e7d2ace23e487cbae339baa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4542efefe8f666234f64f3301bc1ef7
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1156], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_75e983b492c1427fd1695e8b1cd8c424(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 1156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a3dd07821182203eaec9cfffb351590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75e983b492c1427fd1695e8b1cd8c424
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1156], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ae050a5234f3f9a130e14111cddeaa90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 640, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2304c13e2fa940484268f1dd4eebadc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae050a5234f3f9a130e14111cddeaa90
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_59e04fe6a558f2e5f4a8621c12f2cd54(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 128, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f336a02d05a5e269e435774b96cca67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59e04fe6a558f2e5f4a8621c12f2cd54
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 320], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2e1597e3f9acaefacfa1887f9a06b894(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef6eca8324cb89c425ead0be71e45bdd
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_24fa89f9e243f08ae2c8f4a3926f4c00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb9f70282899345810179c07997ffef4
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_232ad9e18a32d2feca61f4f85ae4050b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dc2635fc07050ef006a0bdd589c884e
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c306ebccb1028ac50fd08ab4b199814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2945ca61ad8e249c717d30897e1fd40f
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4cd76f5ede96d6363cd47d42235b7000(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_765dea6d119b72d25e404af98eb9718a
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ef34883d20f522e17d7e43049b0030f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_04a32121f732ee86ca943684e340c206(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef34883d20f522e17d7e43049b0030f4
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d025e6d41ed33c1cf5f5392c03e134f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_407e86043743db80b886354782c8f9d0
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 3136], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8ce79ba9729dde7624b9d55bc2fa90a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55cc5bba00b9cb3fdc64e55f67d86fe8
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b23f30da9c7e03e77b63958f81b8e041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9c3194fb9648928281c02df6c81aca6
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_96638f7ff96debaf0768f3a63848e10a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 361], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f894624a75e067edafe9cd15ef9ec0df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96638f7ff96debaf0768f3a63848e10a
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 361], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6a1d170e58c557fc317d00b260768bbc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 361], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a40a6a3985613892a0a21792d19969a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a1d170e58c557fc317d00b260768bbc
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 361], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_051734f99c6c33b33858c391dfbcc053(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_343f7d8725b39b02a9cffe0ced47c44c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 676], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cac556089bd23fb787c69455bf8bf970(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_341bbfa11050d0985660b1721a2a2775
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 676], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c8720a1a95dd8cd9358aef827dc3a5f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 196, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a8eb833c784fc5a495432939e6557e2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8720a1a95dd8cd9358aef827dc3a5f6
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_547ca365aa33ec018d8aaabadc95f9f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50e179c887d95e472bab06e86b01d414
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fb9eb20474bb4e4f7d0f85180b7cb4a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 384, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b1317a9f83134827e3190cf9884e4089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb9eb20474bb4e4f7d0f85180b7cb4a6
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_07bfc4e5f5542767c9d6eaecc9872d75(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 2, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5dd14be3064b4c7e75f9540b77229a5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bfc4e5f5542767c9d6eaecc9872d75
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8cbc16abd8a3736cc9be4a9e2478f813(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 12, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ace5d8565b6c141703d49daf804b468c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cbc16abd8a3736cc9be4a9e2478f813
    def get_inputs(self):
        return [
            paddle.uniform([11, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_834a01a99007e20f7d8b618a2f8e26f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_595f21ae57133de93b9d170c3981407a
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 3, 4, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_43e558954f8dd84d2ef1e1d2702f8705(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_768c972b03a389502e72873a3e86a2fc
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_457280032f7128c4b2520b4400c69a4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccf813327c1df683ffd36b417520bc5d
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1e7a7e5b4e1ac764f7cc52f270e65538(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_540e613b338922b91701499f99b9e168
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 6400], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e8cf598e584f9548ca50f937fe44a818(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 6400], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e712a130c2ab02278ad087222748c9e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8cf598e584f9548ca50f937fe44a818
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 6400], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_04a32121f732ee86ca943684e340c206(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef34883d20f522e17d7e43049b0030f4
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5347f41a2f1879a82c28a9306eb3b1c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1bdc423af734e1ea8207f8a0f7eb1aa
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_dc1e24823ee25b0d1804e4cd8855be49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 900], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_402026505eb7f5cbf24c2e917e766fbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc1e24823ee25b0d1804e4cd8855be49
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 900], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_30348972b680af2d50ed51e5652735c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 900], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f2507dcc6b807adde480706809eb79d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30348972b680af2d50ed51e5652735c5
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 900], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e7daff024cfaac9ff6127c6681996bd5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[44, 8, 288, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1bdbdc20efd74780dc73049bb6d1e8db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7daff024cfaac9ff6127c6681996bd5
    def get_inputs(self):
        return [
            paddle.uniform([44, 8, 288, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c9cfdff6d2027ad44ce5f1bb978aa35f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1, 2, 24, 12, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1100cd618f2159034f01f9eaf1ee4dbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9cfdff6d2027ad44ce5f1bb978aa35f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 2, 24, 12, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b0ac7c729582f2369dd5dd70416da9a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 8, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de8cfdea7ce53915d9d6d2b14d675b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0ac7c729582f2369dd5dd70416da9a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f16a1270a6c0a5a26bcb0adca1c6522b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 2, 8, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c75ef7b2dfe1fff4e9e2b5ebd15a829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f16a1270a6c0a5a26bcb0adca1c6522b
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 8, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_58f468c11d734f8dd9fffb76323389cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e2c3ec124340a845cad764a4f4c9c4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58f468c11d734f8dd9fffb76323389cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cea21b67845ee94976c1917716f67ffc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_884679eb5d7e1c223619169c1907f115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cea21b67845ee94976c1917716f67ffc
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a8eb833c784fc5a495432939e6557e2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8720a1a95dd8cd9358aef827dc3a5f6
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_547ca365aa33ec018d8aaabadc95f9f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50e179c887d95e472bab06e86b01d414
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1317a9f83134827e3190cf9884e4089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb9eb20474bb4e4f7d0f85180b7cb4a6
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5dd14be3064b4c7e75f9540b77229a5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bfc4e5f5542767c9d6eaecc9872d75
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ace5d8565b6c141703d49daf804b468c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cbc16abd8a3736cc9be4a9e2478f813
    def get_inputs(self):
        return [
            paddle.uniform([11, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_64f1c49b4c4b7a7267ce1e66799e24c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cfe7ad2aae62c72b982f787c8d156971(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64f1c49b4c4b7a7267ce1e66799e24c6
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 12, 12], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5e3a4e13cd1072c830b56120be61e93d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4590d695db9f8983f1bc26345670654
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 3, 4, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee3ac876ea1956637e08da2ab2b76ef9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c6da0f0264d7892c1de7fb8d258e236
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f2f0176f9eeb17731d951249ed3b46e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_138bae9e0cfad9b1e553a4cf0b6144a3
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f98830438f983e45bda5183763ee1c19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8d9b63e2c91b585c5fed809877a683
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 20, 128, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bf4eaee4be9a6a8de201abe8012687e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_892f8c36460cc78452c2435bef822a26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 40, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fef519db7a9cf4aaf76fba81a282032e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 3, 6, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a881fce929bee31cf205db2e7b963976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fef519db7a9cf4aaf76fba81a282032e
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 3, 6, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a548a2aa221ad914f7023dc74380aa27(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1174, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6bf9b32c453a0d67390b5d626a07bd43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a548a2aa221ad914f7023dc74380aa27
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f9e54586d17ae3b643e97d6f3e428c8a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1174, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_58a85767ce29bd130393ac56db771e28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9e54586d17ae3b643e97d6f3e428c8a
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ac85a410e97f8f79b85c60dcf97b0f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dad0e76957d256859563cd800db89cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 400], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_db91778fdc4adc3b14ba7b400c495576(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 400], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2b610d7461189181979a4c24b6cfb985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db91778fdc4adc3b14ba7b400c495576
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 400], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_db523dc799dcdb32406127300fde26e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 400], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e01993183f4c0879581310ba64229df0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db523dc799dcdb32406127300fde26e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 400], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cd471f5a8daf7f77a06ef2ddde2f493c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 225], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d164ce40076d048b5386ffae87ab5a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd471f5a8daf7f77a06ef2ddde2f493c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 225], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_57373ea12f862db5375fc3811a66cc15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 225], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_48ed5bcf023ceae0e781bbd3bb9e1b1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57373ea12f862db5375fc3811a66cc15
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 225], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_795c512a7529c4b1b1c47810a8e3ce73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bfbd9f96dbdee65f2aa436b7e2555021(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_795c512a7529c4b1b1c47810a8e3ce73
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1053c0ff64f9d89fc24745add30b0f83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 4, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b3509e684d38d87fa8a0591bdf9291a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1053c0ff64f9d89fc24745add30b0f83
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 19], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6a075d50ad9c4bf912cb6774e8c1b7b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f6c6361dc5654339d27ac533307ae9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 169], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1214c097f1a2d9e402428cebe068787b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 76, 169], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ff78b996541f39cfffeb5948a7dda475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1214c097f1a2d9e402428cebe068787b
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 169], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1a37be9ee1195b7c221974521c0531a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_289befdd18cd8a36bda6d55175c8d723(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a37be9ee1195b7c221974521c0531a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f80d70bf5d4fbb5c1e419ffd8c79350d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_385fd542618f80dcbfc9567baaca9e40
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f1d18e4d693cf56500edc85c4cbae66a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_867c60590970bc95bb0e4fb82b534813
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_581160e7a0244e40c03be34e1312b7eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79155854e8f8ff91e499fb119522bbdf
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_397302b126ccdacea9dde585e1c4345d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1960, 16, 2, 4, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d5a7751b8c1470fbba3be7565d0532eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_397302b126ccdacea9dde585e1c4345d
    def get_inputs(self):
        return [
            paddle.uniform([1960, 16, 2, 4, 6], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7c15f52b3d29ea5aa27b47b54514c4e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1960, 16, 4, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_76e02c9428111567d7c2fa0ff734a111(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c15f52b3d29ea5aa27b47b54514c4e1
    def get_inputs(self):
        return [
            paddle.uniform([1960, 16, 4, 6], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f9c074e871e9dd2fa48802a463e7d808(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1960, 4, 16, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_63d8a06c19064efcae27056000aea02c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9c074e871e9dd2fa48802a463e7d808
    def get_inputs(self):
        return [
            paddle.uniform([1960, 4, 16, 6], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a84564e60d025ac9b5b2ab5397fd4962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_581351d2c9f222d4e806d2f588dfd829
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e84af7c453015be1be4a52c10edbf23f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6f4128b5aa91d80541ab4f5fcb6f9e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9ec09d020c001820261942dcd6855aaf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 200, 272], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_627bb74150373bee7915c7be262d6f7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ec09d020c001820261942dcd6855aaf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 200, 272], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_623ca32d6aac6840c4013ff7ca5391cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 100, 136], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_feacfa936f60289f624ae06997342a96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_623ca32d6aac6840c4013ff7ca5391cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 100, 136], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f1b57fe7b1885402e6716364ccf46eb1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 50, 68], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_facbf313beed795204024a054d20ff6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1b57fe7b1885402e6716364ccf46eb1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 50, 68], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f626516335316c95f009dbe5c63580ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 25, 34], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aca0d9340fe69fe689576efa8afd415d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f626516335316c95f009dbe5c63580ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 25, 34], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0539fdda9fa45be3a2ca503eb32354f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 13, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6da917614311bf6effa2aa1e6a4046c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0539fdda9fa45be3a2ca503eb32354f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 13, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_53d31089c17ed3772e935ab828695f80(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 200, 272], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d69ae62729167078709681a05199e447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53d31089c17ed3772e935ab828695f80
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 200, 272], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_114b14b31ebd06b19b373e9cdad89fbf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 100, 136], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_90198fa39660acfa8f950c1a33647842(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_114b14b31ebd06b19b373e9cdad89fbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 100, 136], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cf9a81703bfb55a60dc625caddf0e244(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 50, 68], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c8fa42c87d5ac4c466bb18056fd00a0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf9a81703bfb55a60dc625caddf0e244
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 50, 68], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9e72c184267002ddae13a60394f17ad1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 25, 34], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f11c501f0e69c31ed423b367d03245b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e72c184267002ddae13a60394f17ad1
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 25, 34], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3c69fc901782220a6e8d3f827023c5c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 13, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f3f9fe3d87472bd153a73fb774c65622(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c69fc901782220a6e8d3f827023c5c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 13, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_976f2ab0cbc1cb1de6413db5cc26e099(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 400], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d5bec9aa555a9102fa2b31762586ce1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_976f2ab0cbc1cb1de6413db5cc26e099
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 400], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_207b632120344fd362c42eaadfea9f4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c51269703391cb3c9db022ed4a74d00
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 400], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a7d1b0268d349ee63c28b2a6f5f09263(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9874f711042e2bfd107011a52d643e65
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_91ce96f4f675c6b3875a1bad1955a935(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_736429e2d6de83812117a67bd8bf0eec
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4884a40b93594d469e258132bdbdd56f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c9267bec5ee55f9afae9c6d978b6d09
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_aa64eec84a2c66d1fd8342a25636c318(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 192, 25], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b3dca762cbbfbbea29adaafaeade39f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa64eec84a2c66d1fd8342a25636c318
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 25], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_02eee51af78e7e19f807057f2847e0f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1247e5974bcf831b85bff78035225f53
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 7, 1, 7, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cb63ddee5d3ce032d7d3a8da982321ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92dedaec40aad3f7c347bb59c85ff802
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 49, 3, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f9e1726b0cb3bef625078eb72be1e6fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5956615907214bfca4df75c5c979ad5
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f56584f9f9e80530fc61f4e2165312c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2112, 2, 96, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fd6d4b4b99ad11ba2d245e8867166d07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f56584f9f9e80530fc61f4e2165312c1
    def get_inputs(self):
        return [
            paddle.uniform([2112, 2, 96, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f1472c411c7d21a4c2b4b8bda078a569(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 96, 1, 1, 96, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f6a1ccea8b999c49e7dedec598cfd0b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1472c411c7d21a4c2b4b8bda078a569
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 1, 1, 96, 48], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0c5724184a7b3f12405d87672c25b77d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 784], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2b238fe74c3906d7ad43dd6ec9b58652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c5724184a7b3f12405d87672c25b77d
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 784], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9ea53b525a5671f10a524920204e182a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec50c8425c07efb66eb45659e5e61cd3
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 784], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_15155b54080fdcbb2d74186114d247f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[96, 4, 96, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ff642881438e4b62002ac6ecd16cbe52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15155b54080fdcbb2d74186114d247f5
    def get_inputs(self):
        return [
            paddle.uniform([96, 4, 96, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_22b42e72eedbc9f3e976655e247026f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1, 24, 48, 2, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8aee98386c53409e349c97ad5e9aefcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22b42e72eedbc9f3e976655e247026f9
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 24, 48, 2, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7e98c6e3c5948eff2f4fac5ee115f6b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 58, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5ed9ef78394889aac4316a5729cef513(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e98c6e3c5948eff2f4fac5ee115f6b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 58, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a7d1b0268d349ee63c28b2a6f5f09263(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9874f711042e2bfd107011a52d643e65
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_91ce96f4f675c6b3875a1bad1955a935(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_736429e2d6de83812117a67bd8bf0eec
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4884a40b93594d469e258132bdbdd56f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c9267bec5ee55f9afae9c6d978b6d09
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 16384], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()