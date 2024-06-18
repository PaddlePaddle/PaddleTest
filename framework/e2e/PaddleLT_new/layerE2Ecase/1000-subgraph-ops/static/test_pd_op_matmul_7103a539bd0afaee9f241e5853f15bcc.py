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



class PrimitiveOp_1a6848912d3d1803239421afebe55a5f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_533d525e0645bb0e29503e6f6c486dbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a6848912d3d1803239421afebe55a5f
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c9a46af6b80743c86f9a0621dcd11ca8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 1536], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_79baf4642ced18364a96619e373b016a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9a46af6b80743c86f9a0621dcd11ca8
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_826696782712534bb84171bc3495f025(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 100, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab003b122fe889d2254d9d6a4e10f093(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_826696782712534bb84171bc3495f025
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f375d1bc7f15cf3b8e4c9875985bc0fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 100, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 4, 32, 100], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fb8b5a1e69a06fdcb44ea60ca2d1eefa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f375d1bc7f15cf3b8e4c9875985bc0fc
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 32, 100], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_dec2f7ba786fc2c2b0d89ca4a40123f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 100, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 4, 100, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e75e9db2890a366478a293dd0c17c599(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dec2f7ba786fc2c2b0d89ca4a40123f7
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0e8e3c34a16c85a6ebe53aef23c6e178(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 100, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a0d8a17f8775644f5d4a9a354c258a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e8e3c34a16c85a6ebe53aef23c6e178
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a8d0b2fbe162f672c6bb64964189e9ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 4, 4, 7, 7, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4fda3fa7786a0bd9273f3bef57d8e786(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8d0b2fbe162f672c6bb64964189e9ea
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 4, 7, 7, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6c80b49e10b4fc9934cd86fd314d942e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e071904e33566d2ea73a4aeedada659e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c80b49e10b4fc9934cd86fd314d942e
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7fb8085c71d323835b162d2cfe749f0e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1025, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 12, 64, 1025], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c24584d8a4e4955457747c2bdc478dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fb8085c71d323835b162d2cfe749f0e
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 64, 1025], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e2cfdb4e87f6971d3326af0fcf4d950c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1025, 1025], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 12, 1025, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dee25300d5da1ed612fd23fa9a804114(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2cfdb4e87f6971d3326af0fcf4d950c
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 1025, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_53a7cb07669156e56788311f1ad45e5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_088d8352a445e1e635bd49d97747128f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53a7cb07669156e56788311f1ad45e5c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4fda3fa7786a0bd9273f3bef57d8e786(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8d0b2fbe162f672c6bb64964189e9ea
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 4, 7, 7, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_96d13030358781761a1661cd917e000b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c2ca6eb748ec173da9c0028ae262db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96d13030358781761a1661cd917e000b
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3b0b37547c21bf70375542df66aa64aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[15, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3db043c17642ed4bc3d56e37abbe2c40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b0b37547c21bf70375542df66aa64aa
    def get_inputs(self):
        return [
            paddle.uniform([22, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bed1da2d572c3269b59dc6e08ab37468(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672], dtype='float32'),
            paddle.static.InputSpec(shape=[672, 168], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e6f4ffdb015d5e6ea314e2f7b3050104(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed1da2d572c3269b59dc6e08ab37468
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 168], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_12f44cb125e70744868072a1db8187fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 168], dtype='float32'),
            paddle.static.InputSpec(shape=[168, 672], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1850bcb957e5aee4446b1a32ea4ebe41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12f44cb125e70744868072a1db8187fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([168, 672], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fb77ec3dfad1a4af5dd7b22df829bc52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 4, 4, 7, 7, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_96f60e34aee5b264617ab0f094b8cc3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb77ec3dfad1a4af5dd7b22df829bc52
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 4, 7, 7, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_08677397179efd97dbeb14ae955d2f88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b0ca052721045b1348770e695bb5bd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08677397179efd97dbeb14ae955d2f88
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a5278700bd0d2e8d0e029ecea872d714(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b91fa3f7588b6bcd820f29ca6a724d61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5278700bd0d2e8d0e029ecea872d714
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_aaccd6f5e70dfea9a33d20720c8306ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8, 32, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eb0feeae7757bdfaa41562e06992d94c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aaccd6f5e70dfea9a33d20720c8306ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 32, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_743cfdf53e524cda650fd37a40bc3e6b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8, 1024, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_94d6c765a65a678da7bd3353d2376751(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743cfdf53e524cda650fd37a40bc3e6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 1024, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0b0ca052721045b1348770e695bb5bd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08677397179efd97dbeb14ae955d2f88
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_255d35eaee269e4632e9dfba6654a684(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 2304, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4baa668904bc3d0c43eb96e9cb87dd0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_255d35eaee269e4632e9dfba6654a684
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0a41447269f9452fafd914d59e0c055a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ff0f013bfd78f5294f4862ad66aa096c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a41447269f9452fafd914d59e0c055a
    def get_inputs(self):
        return [
            paddle.uniform([171, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ac2589a5036e7dd3951e4533fd59b3da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f4d466d0dddca521d37265f29a706ab6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac2589a5036e7dd3951e4533fd59b3da
    def get_inputs(self):
        return [
            paddle.uniform([171, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7a7742a79733a83469ec013e1b9a0c1f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c0e1315735dc22225f0ebd1cdea936e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a7742a79733a83469ec013e1b9a0c1f
    def get_inputs(self):
        return [
            paddle.uniform([11, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([320, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1171c4694552b1cdce20389f2b382410(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9ef7303fbee17b8e404723ed0686a506(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1171c4694552b1cdce20389f2b382410
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_07526cc8d6c78bbeb4f2ee6e602a20bd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[15, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1aa4c2359b891840a05143df3b54294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07526cc8d6c78bbeb4f2ee6e602a20bd
    def get_inputs(self):
        return [
            paddle.uniform([145, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3368a3f54a0b32d98d1481324bf48fde(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 197, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a5a6a99c733dcc9e19e2a593cbc45b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3368a3f54a0b32d98d1481324bf48fde
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_51b3b0b55be7c8cf53070b68cf76b7d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[54, 3, 64, 197], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_34d1de9cbe73b720fa267e5a3457f555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51b3b0b55be7c8cf53070b68cf76b7d2
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 3, 64, 197], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5aedeeda2a08e6397cef06b8a3691dab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 197, 197], dtype='float32'),
            paddle.static.InputSpec(shape=[54, 3, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57105f12683e51b9f17bb33e925c97a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aedeeda2a08e6397cef06b8a3691dab
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 197], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_055bdf71affbe7c031cb2287669f30f0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 197, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc9f2635045a39f92b9bb7ad39f57bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_055bdf71affbe7c031cb2287669f30f0
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a421fb27353afd32d9cd3546b106f227(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 2, 2, 7, 7, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5ed1eb3c02e76c80f196c605e175d753(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a421fb27353afd32d9cd3546b106f227
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 2, 7, 7, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bc9c68e94c153276787f75d3c3af386f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 2, 2, 7, 7, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d06bc4262d620fc2cbe2725872ae3c40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc9c68e94c153276787f75d3c3af386f
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 2, 7, 7, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a2e8cf438795f6c06d9d8ebbe0bf2147(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 784, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f65c295337fa6101aa02ed62148cff53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2e8cf438795f6c06d9d8ebbe0bf2147
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_96288a25145efa7447b500a4ae2ae028(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d0b41bb373e47cb494a7cd350181e61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96288a25145efa7447b500a4ae2ae028
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_814dd2225bfe3aeec82e8f2aace31132(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 640, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc779845209e738e3686b77afe9fbce8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_814dd2225bfe3aeec82e8f2aace31132
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7566ea3d57b8f4d387ce60ebbe6624ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 640, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 2, 32, 640], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1208723d804ca6413e9507e1f8d07eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7566ea3d57b8f4d387ce60ebbe6624ab
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 2, 32, 640], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b9ad3293abcb8aeb366dd8a2efbca664(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 640, 640], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 2, 640, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a855a2e8b585cde62963ecaa7e56b00c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9ad3293abcb8aeb366dd8a2efbca664
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 640], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b9ab4debacb8d079f701e28937804f47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 640, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_98595126ebbc51ba643a306c5eadf838(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9ab4debacb8d079f701e28937804f47
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f65c295337fa6101aa02ed62148cff53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2e8cf438795f6c06d9d8ebbe0bf2147
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d0b41bb373e47cb494a7cd350181e61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96288a25145efa7447b500a4ae2ae028
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9e505cd0171a73bc4b5bfba87f5867d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4504c4f7ddb2ec44434dfa7964b6e6b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e505cd0171a73bc4b5bfba87f5867d3
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_05cbb7b4e778225ccef30208acdb8a22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c5a5414f90f6df76ffc7e012cd73c491(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05cbb7b4e778225ccef30208acdb8a22
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ddfa228cfaacd7ffe628a2c2ae74c0ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 8, 8, 7, 7, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6edccfe0d906a85495c93ccb49e49ee6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddfa228cfaacd7ffe628a2c2ae74c0ff
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 8, 7, 7, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b228aa0926ef58da9a91895398c5d90c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_22ccd2c9b965038acd6f9d63ba40ad03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b228aa0926ef58da9a91895398c5d90c
    def get_inputs(self):
        return [
            paddle.uniform([22, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_925edaef74e7d9bf9472c71d59858f4f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0c9ffec61d8c389b69f13efb944a2b78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_925edaef74e7d9bf9472c71d59858f4f
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bc7a1d76b1d46779ebaff576ae5a08bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21, 16384], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 16384, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf7b2c5d33ba281f216d77c23b5f3fb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc7a1d76b1d46779ebaff576ae5a08bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 16384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 16384, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8daf229956d353ad23077da96e98d0c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 200, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c632876e5b8632acab1a1a74a3384cf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8daf229956d353ad23077da96e98d0c8
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a8fc5ac177feb309de9bccb4f6349b21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 200, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 2, 32, 200], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_061e68e275094a941b76d7cdb0778803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8fc5ac177feb309de9bccb4f6349b21
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 2, 32, 200], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2ac9d540f61e15e788af3107c917e7b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 200, 200], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 2, 200, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_14e1685c9bf9dd75e4cdcf18b4956569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ac9d540f61e15e788af3107c917e7b5
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2acfc935f0366deb4981ba167f656cdb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 200, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9fcead614f82b13362ff99cb9760a883(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2acfc935f0366deb4981ba167f656cdb
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8cca3497c9f5635aae9dad563dd7a0d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7f3e38d93289a95f73914fa3892d402b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cca3497c9f5635aae9dad563dd7a0d9
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b0d33df3f68ae41bbd94de1e59b80100(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5a1f514c0058cd0f546f3aa92784a7cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0d33df3f68ae41bbd94de1e59b80100
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e645a720fe263c8b2246553bdb61fa9c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_37331fe8052b1b10abbe86e20c5941af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e645a720fe263c8b2246553bdb61fa9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d914cb23264616c40530d478dc3a4c54(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1536], dtype='float32'),
            paddle.static.InputSpec(shape=[1536, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_26bd94e2dec9767935db2632283f602f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d914cb23264616c40530d478dc3a4c54
    def get_inputs(self):
        return [
            paddle.uniform([10, 1536], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1536, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b546cdf23ca1dde703680da387cfa84a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2048], dtype='float32'),
            paddle.static.InputSpec(shape=[2048, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bbdcb54aeea5b91db3f67e4876268f90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b546cdf23ca1dde703680da387cfa84a
    def get_inputs(self):
        return [
            paddle.uniform([10, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_21d2dfe41a46a8b0a7af0ff57de18eef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 784, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e7ce2c18561687f0ddbfc9f2d9bcb116(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21d2dfe41a46a8b0a7af0ff57de18eef
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4f27ab9b93e77c852a668b6fd1d024f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a935f34daaea4bd08e3efb16e37c1302(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f27ab9b93e77c852a668b6fd1d024f6
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_52da0d266faaf25e692b2aefcbca3913(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dd85d6eb0d9e8a99ee3138c9e5347e02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52da0d266faaf25e692b2aefcbca3913
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_370350f002af61048e0359916c4b7d54(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef2b59f17785264da630b2aee088a48c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_370350f002af61048e0359916c4b7d54
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3364730273fb69feacf15ec365b5a455(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3eab449c9b72255d67a0322ddada6b9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3364730273fb69feacf15ec365b5a455
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d646e691a8e02157f64d3e9287a3661f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[15, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4989d39dced489f77ea23027916475ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d646e691a8e02157f64d3e9287a3661f
    def get_inputs(self):
        return [
            paddle.uniform([10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_49f66a1bb4a0b9d71193a065e6a02f00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 3136, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2b81af14a1e1233930e755daacb518f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49f66a1bb4a0b9d71193a065e6a02f00
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bc256b8a879005ad7db9d33f98ccede7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9a9c22122416898c1d7972a15148a51d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc256b8a879005ad7db9d33f98ccede7
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4c62c8f2379e2227f610bdc1a9efc9a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 4096], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4096, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d10a6c9223c81e2ec97bac9c5ccc7643(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c62c8f2379e2227f610bdc1a9efc9a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4096], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4096, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bcb07b100ddd9ee57d5270eaa26d7ee0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4f4258c1099b9bab06bafde8f7d47f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb07b100ddd9ee57d5270eaa26d7ee0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_75620f0a333df7d041d7922170770987(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21504, 1, 91], dtype='float32'),
            paddle.static.InputSpec(shape=[91], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e3d5a3d567466b612967cd332ba9664(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75620f0a333df7d041d7922170770987
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 1, 91], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([91], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_95b606940e7b0302ed7a72599f5f4b0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4312, 16, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39f1aca979240974c987483b778578e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95b606940e7b0302ed7a72599f5f4b0b
    def get_inputs(self):
        return [
            paddle.uniform([4312, 16, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 48], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_529ee53acae9a7b6a46533807ce2e9ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4312, 16, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bd884929dbcf02f68ff9e5a7620b09ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_529ee53acae9a7b6a46533807ce2e9ae
    def get_inputs(self):
        return [
            paddle.uniform([4312, 16, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_74fc7a88a431358d204459f3828b3992(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 1, 7, 7, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d9c6db21c89d005f59f5fd5f990aa2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74fc7a88a431358d204459f3828b3992
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 1, 7, 7, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8ee7b4e00f6e194b8afee3d6abf93e60(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 40, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 6625], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9cdd2008a32f17b9847540876a87344b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ee7b4e00f6e194b8afee3d6abf93e60
    def get_inputs(self):
        return [
            paddle.uniform([10, 40, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 6625], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_558b00efbd7870c0454b69f1a319d9d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 160, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2fc3fc668ec394bb68bfeb75738ad0ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_558b00efbd7870c0454b69f1a319d9d0
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_59661d97c7726abb9c17cbbadf386186(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 160, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 8, 32, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a24b882e96d88fa7c675560f765f4608(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59661d97c7726abb9c17cbbadf386186
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 8, 32, 160], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0e39ba1f4afe2a25444a14f20cccee28(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 160, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 8, 160, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f0d6a7bc224181d06436782cd5f6b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e39ba1f4afe2a25444a14f20cccee28
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_87e0a71c482a6a9b6c55d5a8aeb985bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 160, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_082e1aa9158141de388ae007bd7ca67b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87e0a71c482a6a9b6c55d5a8aeb985bf
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fdce1f16e9dbf7a61d6f87c0e2807b6b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960], dtype='float32'),
            paddle.static.InputSpec(shape=[960, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0fdaa6788e0b33e7935d02928e940e53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fdce1f16e9dbf7a61d6f87c0e2807b6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 960], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([960, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bc6c3ee870b8cfb2a29195199010d83f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 960], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b27db2aacc944828d07511d64225fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc6c3ee870b8cfb2a29195199010d83f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 960], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2dc34e5f120c753e7b52600fe3f77420(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 169, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 2048], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_98040401fa12e6fb73e77395f2174c29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2dc34e5f120c753e7b52600fe3f77420
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1024, 2048], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_79443f6d1c2beee9d8c3e7f2c6dd0af9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 169, 2048], dtype='float32'),
            paddle.static.InputSpec(shape=[2048, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_517a81b29061945aeaa997d963eda6f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79443f6d1c2beee9d8c3e7f2c6dd0af9
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2048, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e93642f386a210f4fc50bb03a2b44104(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 624], dtype='float32'),
            paddle.static.InputSpec(shape=[624, 156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f3ad9bb2d350db337213ca1f506817ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e93642f386a210f4fc50bb03a2b44104
    def get_inputs(self):
        return [
            paddle.uniform([1, 624], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([624, 156], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_53e8a8085f483cd1f6734e462d9b93d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 156], dtype='float32'),
            paddle.static.InputSpec(shape=[156, 624], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_850d01fbb783f2f8efae165e07a9755f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53e8a8085f483cd1f6734e462d9b93d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([156, 624], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_96f60e34aee5b264617ab0f094b8cc3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb77ec3dfad1a4af5dd7b22df829bc52
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 4, 7, 7, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_830b1da57a82fb532b63e82e7557b207(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_380659f33db6e048354b42111e4d9ca1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_830b1da57a82fb532b63e82e7557b207
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d20da735e52105fd6b4d39f7a76e0c66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[15, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e9b90b8a40aab88de915952cbeb65e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d20da735e52105fd6b4d39f7a76e0c66
    def get_inputs(self):
        return [
            paddle.uniform([171, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a099b61500527ba2f9db68295252cdf1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_47e0f69cb8e1caf2a5fe243a257f5818(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a099b61500527ba2f9db68295252cdf1
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_981791f248bb06322df9a2532976f8e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_78a79ef0a77a8d3112b8e45f384b2968(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_981791f248bb06322df9a2532976f8e8
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e35525c7fee13b8025fe834922178e11(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 320, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_095c0797933f2317d197fe724c811082(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e35525c7fee13b8025fe834922178e11
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_220fc74b353ba963fb102827f10ed106(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 320, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 4, 32, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ec11ef75fca46e67b814bd248c742cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_220fc74b353ba963fb102827f10ed106
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 32, 320], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0b87fc542d727b5a442058625834cbf1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 320, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 4, 320, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c651080b9917d84b92c4bd685c5442e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b87fc542d727b5a442058625834cbf1
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_84a102ae76f24d06a8a28b2af18c9c14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 320, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a2d3fb15498b45fa3f944a1a418d874f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84a102ae76f24d06a8a28b2af18c9c14
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_dee4577010774ed9a7cab09e777dd25b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7007040da1df5e7dac15ec76e83de331(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dee4577010774ed9a7cab09e777dd25b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f4338fc0c69582001b6b906a50b95ccb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a84bade421d0a0dffdef6acf42118c71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4338fc0c69582001b6b906a50b95ccb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a1e87236b740b11c6d930b750bb79a2a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[36, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d5164226195f3fdfe6ec79eadc5a6cd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1e87236b740b11c6d930b750bb79a2a
    def get_inputs(self):
        return [
            paddle.uniform([145, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5e20630b21968cd71831fd170d06bb70(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[9, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_31a6693a43ce015e6cd2bf782dac6ac7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e20630b21968cd71831fd170d06bb70
    def get_inputs(self):
        return [
            paddle.uniform([145, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_47e0f69cb8e1caf2a5fe243a257f5818(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a099b61500527ba2f9db68295252cdf1
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_78a79ef0a77a8d3112b8e45f384b2968(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_981791f248bb06322df9a2532976f8e8
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3b06119602e56c152c60bee9c94886c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[390, 3136], dtype='float32'),
            paddle.static.InputSpec(shape=[3136, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23d88f8e55462ce8dc69f2ba71edeb28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b06119602e56c152c60bee9c94886c1
    def get_inputs(self):
        return [
            paddle.uniform([390, 3136], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3136, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a3d609ad079978db076d7a315ddd5b51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[390, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf1e6e5cb12bdde84b135775fc6564de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3d609ad079978db076d7a315ddd5b51
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1024, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5162b7e3573bdd2f0b417fa5fa18c8ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1eadc4fec1633b673d6fdb2990fe877b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5162b7e3573bdd2f0b417fa5fa18c8ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c7e99ed851bf2fda824479676ee92288(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1da2045f8ebd0a02f0de006dc9de344(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7e99ed851bf2fda824479676ee92288
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e4138edc2063e178d976f146c00140e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8, 32, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6a3c03d65354824426c8dc595986d84b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4138edc2063e178d976f146c00140e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 32, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6744f2e3f9cbed8f5e22dc0124e6c703(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8, 512, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8124d66dd99265b4aba62abe0dab7271(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6744f2e3f9cbed8f5e22dc0124e6c703
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 512, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1eadc4fec1633b673d6fdb2990fe877b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5162b7e3573bdd2f0b417fa5fa18c8ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8f201091710cf1b5d33779193a44660c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 2304, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c1ffd3c230e89154be23f9e47bdc9692(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f201091710cf1b5d33779193a44660c
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9a7b1a00391e56611a17112b3b2efd3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72], dtype='float32'),
            paddle.static.InputSpec(shape=[72, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60d0a92d39b6dbbe06325d9324df3317(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a7b1a00391e56611a17112b3b2efd3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([72, 18], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_58c572be2025e36d612402bad799cb3f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 18], dtype='float32'),
            paddle.static.InputSpec(shape=[18, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0382c3e77a380a2ed8f7323809c9aabf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58c572be2025e36d612402bad799cb3f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3817678689956665, 0.0, 1.1630475521087646, 0.8850731253623962, 0.0, 0.0, 0.24910926818847656, 0.29176172614097595, 0.03496110439300537, 0.5644372701644897, 0.0, 0.3801472783088684]], dtype='float32').reshape([1, 18]),
            paddle.uniform([18, 72], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c68e0a3e824229b7f491df33e77df2af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 704], dtype='float32'),
            paddle.static.InputSpec(shape=[704, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0abdc6cfd9ee3592f07758b80a15e34c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c68e0a3e824229b7f491df33e77df2af
    def get_inputs(self):
        return [
            paddle.uniform([11, 704], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([704, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a60879f2e81184da55af1653fd7e03a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 198, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_706f6d40b88fef8ccf5cb05ea32881cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a60879f2e81184da55af1653fd7e03a6
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2a3ebd92f4232af490f5ee6fe7b3490e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 198, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[54, 3, 64, 198], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b231dfd45b61d2baba13a8112c3ac4a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a3ebd92f4232af490f5ee6fe7b3490e
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 3, 64, 198], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_367d464a17c3cb1c7e1195ceeba87d7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 198, 198], dtype='float32'),
            paddle.static.InputSpec(shape=[54, 3, 198, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef75ec9d7299086d051ee9c95f4d000d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_367d464a17c3cb1c7e1195ceeba87d7d
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 198], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_66ae6e9d90ba87fac06bf3149d29a9ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 198, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b0884c0b1a12aac9ab81ddc95030b3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66ae6e9d90ba87fac06bf3149d29a9ba
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b03f628d94a40f306b781826bb990867(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_29ab4584dfc7e019650c5f6d6d89fc82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b03f628d94a40f306b781826bb990867
    def get_inputs(self):
        return [
            paddle.uniform([171, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e8b78f943f7cafc42095ae55772dbbfa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e969fbe1fb2af14b33c7a1ae0f4700a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8b78f943f7cafc42095ae55772dbbfa
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ef528ec91a57230215939de004b08220(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 2048], dtype='float32'),
            paddle.static.InputSpec(shape=[2048, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_14bd435399981f7f9c30889cf7bfc44e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef528ec91a57230215939de004b08220
    def get_inputs(self):
        return [
            paddle.uniform([11, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6cce544bc2f6244f89e1277b385a928b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3eb3f5425828f3031d065ff6828f8b51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cce544bc2f6244f89e1277b385a928b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_94331c7ad4712a878a9b450591b08eea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1b8f45e68b5c3130b0ced048077fcb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94331c7ad4712a878a9b450591b08eea
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d60626a2f55b66400f8262ff30c8a937(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8, 64, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b6dcdfdbe5f20cda48e1d2b6f53a591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d60626a2f55b66400f8262ff30c8a937
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 64, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_671a7777d46b2683ffe74a4f9c99c6e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8, 1024, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_709c377f7b160362eb9f38c88635fc9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_671a7777d46b2683ffe74a4f9c99c6e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 1024, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3eb3f5425828f3031d065ff6828f8b51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cce544bc2f6244f89e1277b385a928b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_35b824b09dfc687a2f243cb677cb7ae1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a59c7ba3b2ddfd564e2b5d0e01be65f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35b824b09dfc687a2f243cb677cb7ae1
    def get_inputs(self):
        return [
            paddle.uniform([10, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8e50bcdd42b1be880d0c060c028e8697(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a5e5294f00a0a5f737f740c4d5349f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e50bcdd42b1be880d0c060c028e8697
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7f3e38d93289a95f73914fa3892d402b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cca3497c9f5635aae9dad563dd7a0d9
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5a1f514c0058cd0f546f3aa92784a7cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0d33df3f68ae41bbd94de1e59b80100
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f9399784c89473ac5be50bae66cbf113(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 19, 32768], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 32768, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3dfb1cdf9f0dc1d6b3c65c3fe8d80545(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9399784c89473ac5be50bae66cbf113
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 32768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 32768, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_03c84232f68eb405e94c9809f9d33b10(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 9216, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8872d3de748ad93b2622c69004cf56d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03c84232f68eb405e94c9809f9d33b10
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_eb1dee21b756956e30385d5f1b0b51ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 3136, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3ef8f5afa53e9b72fc825f633e033530(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb1dee21b756956e30385d5f1b0b51ae
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_089cb118a44d34edb3ac535724f982cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b663ec8d2761f770d7f74b5c2415aa69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_089cb118a44d34edb3ac535724f982cf
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4ce5439a2ce2d3081c3549ab197b62a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 8, 8, 7, 7, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e7c6686c73a838c46726f91a004e0439(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ce5439a2ce2d3081c3549ab197b62a0
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 8, 7, 7, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8d3c0fc2900bf5ae9e5496edf4c52fa7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7983fcd03e84433b71bd4a63a6b6460(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d3c0fc2900bf5ae9e5496edf4c52fa7
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_180fffd42f1d11301e37c4f23e1ceeae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 1536], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02d622dc62dab1e4db8ba5da3d431e21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_180fffd42f1d11301e37c4f23e1ceeae
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_03dd170afe2dbc5825ce71301b537163(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_21629e004c39d66d6bbcfbb5ebb07fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03dd170afe2dbc5825ce71301b537163
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0a031fbbefad02404515d314a3f293a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 197, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c5e08ccb238ec5b65519d9102e640bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a031fbbefad02404515d314a3f293a2
    def get_inputs(self):
        return [
            paddle.uniform([22, 197, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_324c54a528ea43c59afc13931d02010b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 197, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6ca39130dd2a099574fbb1c838e90f33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_324c54a528ea43c59afc13931d02010b
    def get_inputs(self):
        return [
            paddle.uniform([22, 197, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7a0d23fe90953c952b919b4e525021a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc694f1b308c40471091d9068d261b53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a0d23fe90953c952b919b4e525021a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8a73e29f4c80d9a73fa21996cf61396a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1174, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 12, 64, 1174], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dea904d9a9067d8301135dc1df0ab33e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a73e29f4c80d9a73fa21996cf61396a
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 64, 1174], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_798546ff56c084696a53193b1701b395(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1174, 1174], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 12, 1174, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dffec3551416615e89b61a2ffa9a360c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_798546ff56c084696a53193b1701b395
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 1174, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_aa5686d68516efd11eaa12bedb108b4f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_611a3d5ea03083c017608634062b52c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa5686d68516efd11eaa12bedb108b4f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4e1cc2222e2fd636e24b1fadd2d8b02f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 704], dtype='float32'),
            paddle.static.InputSpec(shape=[704, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0cb41587230970549cc070d741dd19bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e1cc2222e2fd636e24b1fadd2d8b02f
    def get_inputs(self):
        return [
            paddle.uniform([43, 704], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([704, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_730df679a01c88bb7bf1e3686762d9e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2fc0c5fd712401fcee977dde896a3424(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_730df679a01c88bb7bf1e3686762d9e2
    def get_inputs(self):
        return [
            paddle.uniform([145, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bd0c65c53e7840c22a3ad803e317fada(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_709c9fa203a2c550c008b1f0929bfdea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd0c65c53e7840c22a3ad803e317fada
    def get_inputs(self):
        return [
            paddle.uniform([145, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5fb7a8d58a1408abc4e48307b3a13c88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf5e784c449503166b430cfaa182f031(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fb7a8d58a1408abc4e48307b3a13c88
    def get_inputs(self):
        return [
            paddle.uniform([43, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([320, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c5412cc6a624af54fa2917a37fd0e8ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 198, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_89c2fae79d3a6bca4cf587b1e99efd6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5412cc6a624af54fa2917a37fd0e8ba
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7af62b38b4007d3b02b1af1101812aa3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 198, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[86, 3, 64, 198], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6c7a565cd4f9e986f56e05b7b71aa493(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7af62b38b4007d3b02b1af1101812aa3
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 3, 64, 198], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_de8c6c332f1c495ab88c7ec5270e2b5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 198, 198], dtype='float32'),
            paddle.static.InputSpec(shape=[86, 3, 198, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_433178398391e7d07c8f2fc05795627a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de8c6c332f1c495ab88c7ec5270e2b5c
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 198], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1c7bcddabddd1289eafc08457fdd4d74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 198, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_24a015ebbbd4a8f2ba8eea1bff6d697a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c7bcddabddd1289eafc08457fdd4d74
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c6461e43e822acac243b84c39b6b6732(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[36, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_14fdb0feeaab978b5db3f531bfb754f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6461e43e822acac243b84c39b6b6732
    def get_inputs(self):
        return [
            paddle.uniform([10, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_416c79b3e8d7eaee8bc0756b3e47d5a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[9, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_284e74238d429ac5df6953caefe34dc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_416c79b3e8d7eaee8bc0756b3e47d5a9
    def get_inputs(self):
        return [
            paddle.uniform([10, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6817849dcbd0140ea44dc8cba2ee3e53(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 9216, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a4adcfa2d61dd3ff7dc76309936c0007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6817849dcbd0140ea44dc8cba2ee3e53
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b805623e5384bdf0d4b1c62606c72afa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 64, 8192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d77a919fcaf854b5778f62919b1da8cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b805623e5384bdf0d4b1c62606c72afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 64, 8192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_368ef83cacae843c68f3024fe9baeaf7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 8192], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8192, 8192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_46df84d3a131364637d3ad93d35806db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_368ef83cacae843c68f3024fe9baeaf7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8192, 8192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7c7e5b26cc1d224b16ba0471e8be8ca2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1536], dtype='float32'),
            paddle.static.InputSpec(shape=[1536, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b076f32c22dd2ae01cab92598b26f202(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c7e5b26cc1d224b16ba0471e8be8ca2
    def get_inputs(self):
        return [
            paddle.uniform([22, 1536], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1536, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_533d525e0645bb0e29503e6f6c486dbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a6848912d3d1803239421afebe55a5f
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_79baf4642ced18364a96619e373b016a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9a46af6b80743c86f9a0621dcd11ca8
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b81af14a1e1233930e755daacb518f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49f66a1bb4a0b9d71193a065e6a02f00
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9a9c22122416898c1d7972a15148a51d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc256b8a879005ad7db9d33f98ccede7
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9ae2eb39c27cbafee9f6f213a401aff3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 50, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7150172ac0fc7168aa067a79ae29b740(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ae2eb39c27cbafee9f6f213a401aff3
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0f7070df1297caca30d1b1c2ea5b0b0f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 50, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 8, 32, 50], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9477950f135d98b9a567e61ae40be307(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f7070df1297caca30d1b1c2ea5b0b0f
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 8, 32, 50], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c8fef269caa362c54df57b8a7a10430e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 50, 50], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 8, 50, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8281520d07e1eca2363d9875a780dd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8fef269caa362c54df57b8a7a10430e
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b975387fb190e9d425a197299424b3dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 50, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d72a5c1b659aab27d862ec02eade8e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b975387fb190e9d425a197299424b3dd
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2f8951d5a9a5e65213ccabcce93a6d97(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 2048], dtype='float32'),
            paddle.static.InputSpec(shape=[2048, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_107af3ac349d369def3eb825857b1474(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f8951d5a9a5e65213ccabcce93a6d97
    def get_inputs(self):
        return [
            paddle.uniform([43, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_981e1df439924f6541e81fe96243f56a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[512, 12544], dtype='float32'),
            paddle.static.InputSpec(shape=[12544, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b914f51f56eb3da2fb4671a75490fbc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_981e1df439924f6541e81fe96243f56a
    def get_inputs(self):
        return [
            paddle.uniform([512, 12544], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([12544, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bed2102281eb5898fa854ac50ef0d98b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b6c8c4e61e4116af3f3ce23888a45a82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed2102281eb5898fa854ac50ef0d98b
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1024, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bf020894016617b02bf3135c642c7530(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 1, 7, 7, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6e658928ed384d8172f6b0866cd56829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf020894016617b02bf3135c642c7530
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 7, 7, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3a6ff259185bc157e61c9778c3b394f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 576, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_756ff1c318a1b3f7c0f0b8734330713f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a6ff259185bc157e61c9778c3b394f3
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_30306a47a47560aa0edce7b5830049e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cdd7320c092e3dee7e5057e2843378ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30306a47a47560aa0edce7b5830049e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b87136411668c3a37f876d136de65e5e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1025, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6, 64, 1025], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca7e54c6c27c5b56bbfe20243cd61eb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b87136411668c3a37f876d136de65e5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6, 64, 1025], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_32ef5b36b59a75612157381e266cfead(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1025, 1025], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6, 1025, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc4be59970ebd86aaa377d4c0dcf1d72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32ef5b36b59a75612157381e266cfead
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6, 1025, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_29bc3b1a05e8716a0b001930802530af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c306a89455ce0a052b0da04b26705922(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29bc3b1a05e8716a0b001930802530af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d9c6db21c89d005f59f5fd5f990aa2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74fc7a88a431358d204459f3828b3992
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 1, 7, 7, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d06bc4262d620fc2cbe2725872ae3c40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc9c68e94c153276787f75d3c3af386f
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 2, 7, 7, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3ef8f5afa53e9b72fc825f633e033530(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb1dee21b756956e30385d5f1b0b51ae
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b663ec8d2761f770d7f74b5c2415aa69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_089cb118a44d34edb3ac535724f982cf
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e3ccb2b7e5ec1045a3f0b9aff991e2dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 2048], dtype='float32'),
            paddle.static.InputSpec(shape=[2048, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f13c1b41d7fadf72ec62efd6d014b807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3ccb2b7e5ec1045a3f0b9aff991e2dd
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e8a03693bf071061cf4437e969f51756(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 197, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c6d8decea25cc15171bbd5b23b478c7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8a03693bf071061cf4437e969f51756
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2f932e69024ef09e0bffbe6fb06a1d74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[86, 3, 64, 197], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b247782985f94e4cc32dfea3a84e1df4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f932e69024ef09e0bffbe6fb06a1d74
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 3, 64, 197], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_caa86aceccadddbaf3233cebc053e280(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 197, 197], dtype='float32'),
            paddle.static.InputSpec(shape=[86, 3, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f279f40434a980b4954bf9510a43f273(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_caa86aceccadddbaf3233cebc053e280
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 197], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5694704676f3e22cb293b5d70e437180(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 197, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_10ee2712a96a0aaa34e52a5dcace50ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5694704676f3e22cb293b5d70e437180
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f13c1b41d7fadf72ec62efd6d014b807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3ccb2b7e5ec1045a3f0b9aff991e2dd
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1449add604889adefc44cdc5eabb070a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 64, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e6fa45ae729d9c5e783b95780f2c3c9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1449add604889adefc44cdc5eabb070a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 64, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5db114dfd689e4fbe638eef71affdeb3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 4096], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4096, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4764ff934569c737ef97b94cc0cb113d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5db114dfd689e4fbe638eef71affdeb3
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4096], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4096, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ed1eb3c02e76c80f196c605e175d753(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a421fb27353afd32d9cd3546b106f227
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 2, 7, 7, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e6f4ffdb015d5e6ea314e2f7b3050104(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed1da2d572c3269b59dc6e08ab37468
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 168], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1850bcb957e5aee4446b1a32ea4ebe41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12f44cb125e70744868072a1db8187fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([168, 672], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8700e8a52bc76320e19abf200a12c3b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1280], dtype='float32'),
            paddle.static.InputSpec(shape=[1280, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e20928a1b0e2403fff0c70cb6fdfe91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8700e8a52bc76320e19abf200a12c3b6
    def get_inputs(self):
        return [
            paddle.uniform([43, 1280], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1280, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b4b535d0d27824e2e185dc99416e76d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 197, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3386fda02cc6583063dc70423713beac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4b535d0d27824e2e185dc99416e76d0
    def get_inputs(self):
        return [
            paddle.uniform([10, 197, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_51453ec1b3b151078fe47e5dce2f6114(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 197, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ddf99844bdc2c9d5c0ccd8c28509bd64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51453ec1b3b151078fe47e5dce2f6114
    def get_inputs(self):
        return [
            paddle.uniform([10, 197, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_aa4ced2b4a5b2f46118b2877700fc6ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 577, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_130f5f8f6de10ea900144137bfa95d18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa4ced2b4a5b2f46118b2877700fc6ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_38d302e9aa06a14f59ebed7062a18d25(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 577, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 12, 64, 577], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_48d9a9a37fdc59f6182d217c2e35f066(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38d302e9aa06a14f59ebed7062a18d25
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 64, 577], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_593d7b29fc337010f7dcf24dc3388af5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 577, 577], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 12, 577, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4b3aad0a90759ab397a6e0403a43edd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_593d7b29fc337010f7dcf24dc3388af5
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 577, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ac42427a4943ce96308abbe97275edcd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 577, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ececf63e6cf7c3d24872d45967f2e400(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac42427a4943ce96308abbe97275edcd
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6edccfe0d906a85495c93ccb49e49ee6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddfa228cfaacd7ffe628a2c2ae74c0ff
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 8, 7, 7, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_57a5a8c99f6ac4d7db15e41a1cd33a0e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 156], dtype='float32'),
            paddle.static.InputSpec(shape=[156, 39], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_22b6ba44b561d776d179d6db2ea3b2ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57a5a8c99f6ac4d7db15e41a1cd33a0e
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([156, 39], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6d35b8203e156cfa37b4323a5439432d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 39], dtype='float32'),
            paddle.static.InputSpec(shape=[39, 156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b972f207fed0c0944a98e0fdf0b8c645(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d35b8203e156cfa37b4323a5439432d
    def get_inputs(self):
        return [
            paddle.uniform([1, 39], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([39, 156], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_01c887c23e33afb94b1ce836385bc101(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 8192], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8192, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8a0d1684bceb61ffd994855536f52765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01c887c23e33afb94b1ce836385bc101
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8192, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7ebeddc85f8eb27b69561a941645c0b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 8192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aad6fac8d8ecf1b6db09a171f7431a74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ebeddc85f8eb27b69561a941645c0b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 8192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4504c4f7ddb2ec44434dfa7964b6e6b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e505cd0171a73bc4b5bfba87f5867d3
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c5a5414f90f6df76ffc7e012cd73c491(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05cbb7b4e778225ccef30208acdb8a22
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f13c1b41d7fadf72ec62efd6d014b807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3ccb2b7e5ec1045a3f0b9aff991e2dd
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e7c6686c73a838c46726f91a004e0439(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ce5439a2ce2d3081c3549ab197b62a0
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 8, 7, 7, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_26e13415be7cb1f4b1c84f48caa81b18(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 144, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6c38903e595800b7e0ababe00fd9c682(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26e13415be7cb1f4b1c84f48caa81b18
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c31f9dca70d93ce3ed738f3786b58d21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c7a053eae45ced24e939199764da719b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c31f9dca70d93ce3ed738f3786b58d21
    def get_inputs(self):
        return [
            paddle.uniform([22, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_dfc755c2fcb21588fb8d8aac601a7c45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_16eb335bfb43eeb953eaef4207c573ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dfc755c2fcb21588fb8d8aac601a7c45
    def get_inputs(self):
        return [
            paddle.uniform([22, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_835334b256869c4ce50114d87b366dd5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 872], dtype='float32'),
            paddle.static.InputSpec(shape=[872, 218], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d398a242f62d6007495680ab815fad5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_835334b256869c4ce50114d87b366dd5
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([872, 218], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_041a99c585e4f60e802d588913ad8e7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 218], dtype='float32'),
            paddle.static.InputSpec(shape=[218, 872], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8c3c61157d8a01d76d46fddbabeb3f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_041a99c585e4f60e802d588913ad8e7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([218, 872], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e7ce2c18561687f0ddbfc9f2d9bcb116(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21d2dfe41a46a8b0a7af0ff57de18eef
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a935f34daaea4bd08e3efb16e37c1302(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f27ab9b93e77c852a668b6fd1d024f6
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d398a242f62d6007495680ab815fad5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_835334b256869c4ce50114d87b366dd5
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([872, 218], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e8c3c61157d8a01d76d46fddbabeb3f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_041a99c585e4f60e802d588913ad8e7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([218, 872], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_771c52cb797fc349eae70d6b120508e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f8e63cab982aae7762252bf41b33a91c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_771c52cb797fc349eae70d6b120508e1
    def get_inputs(self):
        return [
            paddle.uniform([86, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f8e63cab982aae7762252bf41b33a91c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_771c52cb797fc349eae70d6b120508e1
    def get_inputs(self):
        return [
            paddle.uniform([86, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fe57b1e85a3f08e4be2ad2b179aaf76c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[92, 23], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5377601ecc55a6b2d5a4087da6419ade(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe57b1e85a3f08e4be2ad2b179aaf76c
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([92, 23], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ea0b81ae654feaffcdc79fbd87d83492(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 23], dtype='float32'),
            paddle.static.InputSpec(shape=[23, 92], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8afb9762c03708a84801acc9e10d7b50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea0b81ae654feaffcdc79fbd87d83492
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.5339933633804321, 0.0, 0.0, 0.7401264309883118, 1.4625356197357178, 0.0, 0.0352882444858551, 0.3148179054260254, 0.7185551524162292, 1.6579532623291016, 0.0, 0.20905748009681702, 0.6209321618080139, 0.9944640398025513, 0.6388636231422424, 0.12219542264938354, 0.0, 0.0, 0.0, 0.8009944558143616, 0.1562061905860901, 1.2091788053512573, 0.0]], dtype='float32').reshape([1, 23]),
            paddle.uniform([23, 92], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e07f57ea93747a45fe702cc3dc9927fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f49212719b95f5aacc3d36a8d7522226(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e07f57ea93747a45fe702cc3dc9927fd
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fea2ea94f227862cb57b2d2f26da4fec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ea42d6fb4eabf11d8b83f1e320765a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fea2ea94f227862cb57b2d2f26da4fec
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_76e168716b6c78993771db7bee728231(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[36, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_525aabf1b2d1fc387055d6ffba3db8d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76e168716b6c78993771db7bee728231
    def get_inputs(self):
        return [
            paddle.uniform([22, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_70b697248090f7220bea36322cf4f0ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[9, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3757814221c010b1887c397b63bd753e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70b697248090f7220bea36322cf4f0ba
    def get_inputs(self):
        return [
            paddle.uniform([22, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_095c0797933f2317d197fe724c811082(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e35525c7fee13b8025fe834922178e11
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4ec11ef75fca46e67b814bd248c742cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_220fc74b353ba963fb102827f10ed106
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 32, 320], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c651080b9917d84b92c4bd685c5442e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b87fc542d727b5a442058625834cbf1
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a2d3fb15498b45fa3f944a1a418d874f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84a102ae76f24d06a8a28b2af18c9c14
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_99f570234699b9c44fd3b3b245cf6695(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 144, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2145d746be250cd3ed2be44736a06f96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99f570234699b9c44fd3b3b245cf6695
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_424e5a8d65593403a004b05c38ae97e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_853ca3a57959a4027977b1f9222d7284(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_424e5a8d65593403a004b05c38ae97e6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a1b2e278d05ca18258367e516a8b524d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc5029cc2c4485fffc0236ac205be7a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1b2e278d05ca18258367e516a8b524d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_30a7b48e56648f79b4e15251ae28c816(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8, 64, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca1e76af8e7ccaa358416d761f1bd1df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30a7b48e56648f79b4e15251ae28c816
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 64, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0d387452472a1964116c857b8a020132(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8, 512, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_10bb00ff7ff2b2bd777ac138de19f1c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d387452472a1964116c857b8a020132
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 512, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_853ca3a57959a4027977b1f9222d7284(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_424e5a8d65593403a004b05c38ae97e6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1a7b06e5cf84b92b72d5cad0d781d2f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6c51d49fe92c7840af0c20e87b071660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7b06e5cf84b92b72d5cad0d781d2f9
    def get_inputs(self):
        return [
            paddle.uniform([54, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6c51d49fe92c7840af0c20e87b071660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7b06e5cf84b92b72d5cad0d781d2f9
    def get_inputs(self):
        return [
            paddle.uniform([54, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f49212719b95f5aacc3d36a8d7522226(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e07f57ea93747a45fe702cc3dc9927fd
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4ea42d6fb4eabf11d8b83f1e320765a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fea2ea94f227862cb57b2d2f26da4fec
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a3ac96ecd448d6677a690af6042d3a94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[36, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c8f6e71a1fa8f58174308e2383c438b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3ac96ecd448d6677a690af6042d3a94
    def get_inputs(self):
        return [
            paddle.uniform([171, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_66f3c17ded8da2fc6790f2f9d8f33d4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[9, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69b6c329c5a9e1592839acc170fd483a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66f3c17ded8da2fc6790f2f9d8f33d4c
    def get_inputs(self):
        return [
            paddle.uniform([171, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab003b122fe889d2254d9d6a4e10f093(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_826696782712534bb84171bc3495f025
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fb8b5a1e69a06fdcb44ea60ca2d1eefa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f375d1bc7f15cf3b8e4c9875985bc0fc
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 32, 100], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e75e9db2890a366478a293dd0c17c599(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dec2f7ba786fc2c2b0d89ca4a40123f7
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1a0d8a17f8775644f5d4a9a354c258a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e8e3c34a16c85a6ebe53aef23c6e178
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ae931a34d6eaa7e88671b285b686e75b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1248], dtype='float32'),
            paddle.static.InputSpec(shape=[1248, 312], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_588f9b9ae2b4a6f9d69c6275097a902d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae931a34d6eaa7e88671b285b686e75b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1248, 312], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1b7a4a437ff863ad8025728464cb9639(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 312], dtype='float32'),
            paddle.static.InputSpec(shape=[312, 1248], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f448f006eafac30ef8033842c478159c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b7a4a437ff863ad8025728464cb9639
    def get_inputs(self):
        return [
            paddle.uniform([1, 312], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([312, 1248], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b20adc57cfbb810298cfbeb95ee99a3b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1280], dtype='float32'),
            paddle.static.InputSpec(shape=[1280, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fb26f48afc2009558b09b7c7b5723928(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b20adc57cfbb810298cfbeb95ee99a3b
    def get_inputs(self):
        return [
            paddle.uniform([11, 1280], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1280, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d5c0bff74b9c4ad70085e4033a8854e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 30], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_544688b7972c20689af29ae9a302eff2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5c0bff74b9c4ad70085e4033a8854e8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 30], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f212535d60ee7a8f48a4d69176424c2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[30, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_101eeb5ebfac6145c01e33f47b59b514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f212535d60ee7a8f48a4d69176424c2c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0, 0.03644591569900513, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1152191162109375, 0.12293130159378052, 0.8033934831619263, 0.0, 0.221456378698349, 2.621079683303833, 0.12440484762191772, 0.9671777486801147, 0.07148033380508423, 0.0, 0.0, 0.0, 0.0, 0.0, 1.008826494216919, 0.0, 1.5459907054901123, 1.3115335702896118, 1.2700486183166504, 0.881662905216217, 1.4873523712158203, 0.0]], dtype='float32').reshape([1, 30]),
            paddle.uniform([30, 120], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b1caf44c36308d0dee3bfcb9bb4a8a90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5542c1c04b867f00d8ee0f09709956bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1caf44c36308d0dee3bfcb9bb4a8a90
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3607f6f76f599b1f22725f00b3b09986(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1174, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6, 64, 1174], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70d0336bc1a650ce640832b4055e4bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3607f6f76f599b1f22725f00b3b09986
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6, 64, 1174], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f599364f295f0a46f2dec3aae65b5dc4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1174, 1174], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6, 1174, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b5e06a666fdf9d8a9f9b6a6c5ca25ae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f599364f295f0a46f2dec3aae65b5dc4
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6, 1174, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_410b8a7445a693bd48a2f123753063bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ceed1b24616826cd4b608ed0f01068e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_410b8a7445a693bd48a2f123753063bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dd85d6eb0d9e8a99ee3138c9e5347e02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52da0d266faaf25e692b2aefcbca3913
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef2b59f17785264da630b2aee088a48c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_370350f002af61048e0359916c4b7d54
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_17512365beb7fe79bacbe0a1baaef4aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8943dbafb25d16a1163b59541d10042b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17512365beb7fe79bacbe0a1baaef4aa
    def get_inputs(self):
        return [
            paddle.uniform([145, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d01b6d5b42169bdea98eaf94def11d41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b1c6a03e8b18afcdc9420aac3b630076(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d01b6d5b42169bdea98eaf94def11d41
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1f93b0305b9e0e63ce542758558e5fe7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_682f9e8f26f83638ce2a9cf2edb42163(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f93b0305b9e0e63ce542758558e5fe7
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2ce8b0fe3a44ec641ac1e341db50528a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49392ab9c32012bae55765118268cbd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ce8b0fe3a44ec641ac1e341db50528a
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c454ff5d49f3b966424bdc4462c7255c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 576, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d3b3d0ecf8c356f898f1dc2abf547652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c454ff5d49f3b966424bdc4462c7255c
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_682f9e8f26f83638ce2a9cf2edb42163(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f93b0305b9e0e63ce542758558e5fe7
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_49392ab9c32012bae55765118268cbd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ce8b0fe3a44ec641ac1e341db50528a
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7983fcd03e84433b71bd4a63a6b6460(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d3c0fc2900bf5ae9e5496edf4c52fa7
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_02d622dc62dab1e4db8ba5da3d431e21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_180fffd42f1d11301e37c4f23e1ceeae
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e01415cf3b3f244a56114291109ad7f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1960, 16, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fbef9f47866d70956d4dbadf047eb839(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e01415cf3b3f244a56114291109ad7f3
    def get_inputs(self):
        return [
            paddle.uniform([1960, 16, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 48], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6fdae45cc8835b18f3a478bf695717f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1960, 16, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09ea889afce9e63d1f1e48fbec9faa16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fdae45cc8835b18f3a478bf695717f1
    def get_inputs(self):
        return [
            paddle.uniform([1960, 16, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e299db34300883d0194165220b0c1c47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_43282a05407798c2974e70d39cf37130(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e299db34300883d0194165220b0c1c47
    def get_inputs(self):
        return [
            paddle.uniform([10, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_15ff64d096f32cc22471ad6a24f6964c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f37341496766e368cb4007ab4d041e0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15ff64d096f32cc22471ad6a24f6964c
    def get_inputs(self):
        return [
            paddle.uniform([10, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d65fa76428a760bd595abc409845bcd4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 25, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 37], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e50f41279c67cec767c7b25c4bc423d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d65fa76428a760bd595abc409845bcd4
    def get_inputs(self):
        return [
            paddle.uniform([10, 25, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 37], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e658928ed384d8172f6b0866cd56829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf020894016617b02bf3135c642c7530
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 7, 7, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()