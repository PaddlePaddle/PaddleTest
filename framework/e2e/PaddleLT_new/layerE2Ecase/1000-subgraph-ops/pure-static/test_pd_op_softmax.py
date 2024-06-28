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



class PrimitiveOp_eddbbea6b5aa4876ca328472e557807c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21504, 1, 91], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_13f40ac789d4dcd9b0421b1a3bc2c931(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eddbbea6b5aa4876ca328472e557807c
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 1, 91], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_16de6bb9eb8d6aa173bbb05fc4c62f59(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 100, 100], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1d78513d89d4e987f70284502e0a875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16de6bb9eb8d6aa173bbb05fc4c62f59
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7493596a88f29fe94e1cd78f574e0e70(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 198, 198], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e57b40195091170fb901f97c18634e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7493596a88f29fe94e1cd78f574e0e70
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 198], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_da5b7b5ed0475ce1040d4a9476318858(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 4, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5616d68c7e1c2a7d9fcf457716446321(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da5b7b5ed0475ce1040d4a9476318858
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_69383c074fb623b7a247acaa0fc722e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 320, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca0f317bfeb7cfd6e9ce5c733147af18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69383c074fb623b7a247acaa0fc722e4
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4ec65e7ddd7e45abbd59653cbfe29bce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 19, 32768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8542e02dbc28ba66c2aed751e3085721(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ec65e7ddd7e45abbd59653cbfe29bce
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 32768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d00052d519e8e20826ec0792a50e5edb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b1dc10d04dc1a38872e4f8614802a9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00052d519e8e20826ec0792a50e5edb
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c6aaa52eb1dfcbad0cfdc5da7f5f075e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21, 16384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8d341a041dc63a5a0880b941045de1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6aaa52eb1dfcbad0cfdc5da7f5f075e
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 16384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b54163bb62cde85c4c961e5960b73cb2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f8a6be9535e8cc2b8501ee83fe9b796f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b54163bb62cde85c4c961e5960b73cb2
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b321258fb4a1810ff845e03fac226bfe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 577, 577], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8eb0fd31f4bc7883657cec74c8401cf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b321258fb4a1810ff845e03fac226bfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e0a8467fad613f2369b07b7331e3eb83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16384, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_605b127a2c651b1d35a24e080ace34c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0a8467fad613f2369b07b7331e3eb83
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_25a1f5b6fd075460f3a8e671071db01a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fbee94aa7f5e80b7a6946fcdad37f412(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25a1f5b6fd075460f3a8e671071db01a
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4c2b8b5cff5e5f3d7b91b909111893ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4327792ca558ebdb9f425c61d28e7822(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c2b8b5cff5e5f3d7b91b909111893ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_02599fa337daa87ebe1c961dd2cde3f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 640, 640], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7aec9ffcb1ef0a528d667aa0c65d1376(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02599fa337daa87ebe1c961dd2cde3f7
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 640], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c831cca124770d205f5c072207323147(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 198, 198], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3c9a57f75ee37d900cf600ab27e4b76c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c831cca124770d205f5c072207323147
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 198], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_319b9ad225d7f5c61e395e02c0d75963(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_501b66a73375752aae3aa369f2c409b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_319b9ad225d7f5c61e395e02c0d75963
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1d78513d89d4e987f70284502e0a875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16de6bb9eb8d6aa173bbb05fc4c62f59
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_37b6492ed4885ef9f667d22c206164a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5daaeececddd0ebcfd9e607bdb134d16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37b6492ed4885ef9f667d22c206164a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_78752b97f56f816b866e6a94cd1c3ade(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0dda8560b2058b95e20d787344767884(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78752b97f56f816b866e6a94cd1c3ade
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_74bf6d93ca2607516ad26236695bbaa3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_28c26da218322e3db142dd5fa85793b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74bf6d93ca2607516ad26236695bbaa3
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_276b5adc4a5f334416ec1e6165cf8ee9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1025, 1025], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_545e96c477193fd34604f80a39fd722e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_276b5adc4a5f334416ec1e6165cf8ee9
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a9db15573dc4de5d0b7a9aac7c1e3594(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a2e90648152fea7e75521eac52d8f20d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9db15573dc4de5d0b7a9aac7c1e3594
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 4096], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_015374c26cbf1cf67db3e5b74e5368d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 2048, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b3a0504657daef084288ee7cfb70a413(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_015374c26cbf1cf67db3e5b74e5368d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7fda9630f20dd44265803092eedf0a66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b4704bb68f0363984653f745eb8debf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fda9630f20dd44265803092eedf0a66
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_33688504d7b2aadadbad92f4f2008834(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 197, 197], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc6961a08b4668572ebc39cebe93b2b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33688504d7b2aadadbad92f4f2008834
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 197], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_676f0ec2978c09b7af457eed8149cf9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 65536, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1ccc131460266037d7904225f3b7bc0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_676f0ec2978c09b7af457eed8149cf9b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ca0f317bfeb7cfd6e9ce5c733147af18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69383c074fb623b7a247acaa0fc722e4
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b8c6d822469959274bc43efeaf60f1a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32768, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02bed291fe426a8716d692d0bda9c17c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8c6d822469959274bc43efeaf60f1a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4e7d3e59f7488559d7e28dc1df951da4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 200, 200], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23975ffc7a931a36bae0c6191aa2a93a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e7d3e59f7488559d7e28dc1df951da4
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 200], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7d158392ff17506d00c7a24ac63dc37a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a5ff4ab8e9df83c65c39d8d1d1ade5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d158392ff17506d00c7a24ac63dc37a
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4c9d339a5408afd91ff6fafa41ec0619(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8192, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6742d69e6fe5952b8246191e39145a97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c9d339a5408afd91ff6fafa41ec0619
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3a0504657daef084288ee7cfb70a413(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_015374c26cbf1cf67db3e5b74e5368d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ed10392be809eb4937b040cbd792b064(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e83e778dbb8394becf92c8fd9c0d94d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed10392be809eb4937b040cbd792b064
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_adeef60f4dfe6adfae0c6fc661199857(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 8192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_99fe8bc5fa7454df245a6a86a4edab31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adeef60f4dfe6adfae0c6fc661199857
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 8192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fe5dd3ac3284154a4395c830533ba415(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1025, 1025], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_06c99ed1bcaa0ad1ab1b00aff251dcec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe5dd3ac3284154a4395c830533ba415
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bb0475bba8e92b6af9ff9197a7ea1e26(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_36f86c86c7285105d25d06940b79b649(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb0475bba8e92b6af9ff9197a7ea1e26
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4704bb68f0363984653f745eb8debf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fda9630f20dd44265803092eedf0a66
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0f2eb6ab4535414bac485a656bfc5b77(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b49ef7d7644478a00e022b7f4ae38dd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f2eb6ab4535414bac485a656bfc5b77
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_789691374fe56a64c47d8662dc452979(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_102908aa6dfee2c317dd6fe2ff85141b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_789691374fe56a64c47d8662dc452979
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7b3a8e4d6388887312a770b5366cd688(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 4096, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_311f5550a9d5550e60f7ecefe5d8371f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b3a8e4d6388887312a770b5366cd688
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_311f5550a9d5550e60f7ecefe5d8371f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b3a8e4d6388887312a770b5366cd688
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_501b66a73375752aae3aa369f2c409b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_319b9ad225d7f5c61e395e02c0d75963
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_605b127a2c651b1d35a24e080ace34c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0a8467fad613f2369b07b7331e3eb83
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6742d69e6fe5952b8246191e39145a97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c9d339a5408afd91ff6fafa41ec0619
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cd576941ecd4c23ff6967d75de97ef44(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 197, 197], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70f99c7d81e18703452ec3396e7995c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd576941ecd4c23ff6967d75de97ef44
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 197], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02bed291fe426a8716d692d0bda9c17c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8c6d822469959274bc43efeaf60f1a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36f86c86c7285105d25d06940b79b649(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb0475bba8e92b6af9ff9197a7ea1e26
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6a67e7215ad9f5caf3d4ea2949bfa93a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_545b05d9b934afdbf5e0a84791900f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a67e7215ad9f5caf3d4ea2949bfa93a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e110b522e52b2984cce3b0dbb35c1de7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 160, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d89e3689b01466bd4d8ead36e17496c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e110b522e52b2984cce3b0dbb35c1de7
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 160], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_af68e04ba1c814218aa32fe29f8a1141(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1174, 1174], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c10b1ce0f522e7de5d2ec1eceeab70c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af68e04ba1c814218aa32fe29f8a1141
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b7987c7a964d2ddeea37566f29fd2597(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ed985815559a824cb161e8ae0d1d17e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7987c7a964d2ddeea37566f29fd2597
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d1b271ddfc67ee9c41d673634f3a2132(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 7, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5fcf1aecd33c6d41061d4dc4a8bfca83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1b271ddfc67ee9c41d673634f3a2132
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 7, 10], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_af7273c4ec9cfc20882f9776a37ce5ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1174, 1174], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5321bebfa7a3bf25a6a3eea5352445b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af7273c4ec9cfc20882f9776a37ce5ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ccc131460266037d7904225f3b7bc0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_676f0ec2978c09b7af457eed8149cf9b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3e288e792959da280b852033ea84fddc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_46006df750bf2bc7eb72bf93df7b34ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e288e792959da280b852033ea84fddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c767601b93d3781caffb0e8cc92aac0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 50, 50], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cce93cc5333d4c6d3c3a23c4595d031d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c767601b93d3781caffb0e8cc92aac0b
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 50], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()