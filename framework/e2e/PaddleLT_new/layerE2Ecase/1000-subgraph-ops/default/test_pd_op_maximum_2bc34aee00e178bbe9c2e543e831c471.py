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



class PrimitiveOp_9b49aa34510522763d72a6c0031adcc7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 576, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e3d79641638b3016d46a72068daab85b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b49aa34510522763d72a6c0031adcc7
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_21723733679cda9e830dc2441459b080(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 23, 35], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a2b2b355b423d41f6c68fff0d8a3a47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21723733679cda9e830dc2441459b080
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_db9bd9300274c0e9cd7ed6222b6dde73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c48945a461a948ca7b932b45da8ac59d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db9bd9300274c0e9cd7ed6222b6dde73
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_cc20a094848a67e308d963ec5296b27d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d8bad7df106ca61acc322e3618aaace(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc20a094848a67e308d963ec5296b27d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_7cf47f1fb1d3a0c430b428dfa0bdca51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f7d1b06be540b4a918d2acf245ad3a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cf47f1fb1d3a0c430b428dfa0bdca51
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


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


class TestPrimitiveOp_10f8e5f4893ade488356f91a8b3846e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c31399550ef202d07270f8162e49c66
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10f8e5f4893ade488356f91a8b3846e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c31399550ef202d07270f8162e49c66
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10f8e5f4893ade488356f91a8b3846e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c31399550ef202d07270f8162e49c66
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10f8e5f4893ade488356f91a8b3846e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c31399550ef202d07270f8162e49c66
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10f8e5f4893ade488356f91a8b3846e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c31399550ef202d07270f8162e49c66
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10f8e5f4893ade488356f91a8b3846e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c31399550ef202d07270f8162e49c66
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9d1a888e2572e2939c956dd8a8ab8d6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fd4d2724f21366174dcc0d4f659c95cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1a888e2572e2939c956dd8a8ab8d6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_236421ce78df60215bd2a7c878c8d700(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e73e0b4e3ba0de55de2c1e175d960d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_236421ce78df60215bd2a7c878c8d700
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_b51da797927005bb38cd1859d817e187(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 10, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_93372466838ad93539cd17c68afa590f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b51da797927005bb38cd1859d817e187
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_e0ffdd5bf7236aed70be849b15ec5d41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 168, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b3c87b322c24101810f39d870848d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0ffdd5bf7236aed70be849b15ec5d41
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_41445b01980185f9e9f7e623d502fb75(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c8fe4b5b59b3da7308318058667baf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41445b01980185f9e9f7e623d502fb75
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_ac1fc699d1242d3cc0e20637a40547a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 12, 18], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f70172e1c6e2848fc88765e9f25221f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac1fc699d1242d3cc0e20637a40547a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_7db5615c85de2947b7aa067640bd6f20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_439ec0ae703f57c31fc70468c06f5e49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7db5615c85de2947b7aa067640bd6f20
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.7196926474571228]], [[0.012517362833023071]], [[0.2099456787109375]], [[0.7811828851699829]], [[-0.9929444789886475]], [[0.8286439180374146]], [[-0.5811107158660889]], [[0.7360983490943909]], [[0.1997171938419342]], [[-0.35771042108535767]], [[0.1584303230047226]], [[0.20604993402957916]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c48945a461a948ca7b932b45da8ac59d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db9bd9300274c0e9cd7ed6222b6dde73
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0d8bad7df106ca61acc322e3618aaace(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc20a094848a67e308d963ec5296b27d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3f7d1b06be540b4a918d2acf245ad3a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cf47f1fb1d3a0c430b428dfa0bdca51
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_de5b3d4393fd2a4dd6508bfc153e9ffd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc08fa7fa17a54ab290952204c65ba60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de5b3d4393fd2a4dd6508bfc153e9ffd
    def get_inputs(self):
        return [
            paddle.uniform([22, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_a60b9ad2635db4b6d73d0d0e42aa152f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f8e4e55269a296a6af038d6393a1a21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a60b9ad2635db4b6d73d0d0e42aa152f
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_30d93ff451a143c1af4c582ec1de15bd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 100, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f16307de183f24f23b857ac6332a10a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30d93ff451a143c1af4c582ec1de15bd
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_028187fcc580e99a097c9c1e40fe811e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 400, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d7a15b5c8cbb0593c0dd567247b1faa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_028187fcc580e99a097c9c1e40fe811e
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_72553509889f68e7b8f7aca55694b8a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40c33fefd99f26132c99e96baf755d28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_40c33fefd99f26132c99e96baf755d28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_791985e0863360b3b07d19d88c268b58(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cdb82f5aeed528ab16f3fe07d4e8aefe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cdb82f5aeed528ab16f3fe07d4e8aefe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_40c33fefd99f26132c99e96baf755d28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_40c33fefd99f26132c99e96baf755d28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_548f5fc874ee925c2086f08c77b8f6a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3024, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fa6507388566db29589374d43e53745f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_548f5fc874ee925c2086f08c77b8f6a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3024, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_19c4d48a4614377a160bef248e7b0e50(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 168], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d4678a5f8c3c5700486fb42e49ef0ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19c4d48a4614377a160bef248e7b0e50
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_361104abfdbefc900fe5bf4c5f7622a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 672], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f86ac4a04aace26c9eedc8ad4db9e33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_361104abfdbefc900fe5bf4c5f7622a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c266248267552989ecd0b9299365d7a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c266248267552989ecd0b9299365d7a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ec83b15deb58f8f3029f82ac101259e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ec83b15deb58f8f3029f82ac101259e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c266248267552989ecd0b9299365d7a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c266248267552989ecd0b9299365d7a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f98d759b9838c927b7c7c2ba09ac2bd2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4725, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_041a3b865d546f3406a908ec915a9af0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f98d759b9838c927b7c7c2ba09ac2bd2
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4725, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7b3c87b322c24101810f39d870848d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0ffdd5bf7236aed70be849b15ec5d41
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1c8fe4b5b59b3da7308318058667baf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41445b01980185f9e9f7e623d502fb75
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_aaf550957e26b3fa1ff256401484b75d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70cb339b3f7a24cedc26cc1de6728434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aaf550957e26b3fa1ff256401484b75d
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_3005ea7f40d57fbaf1827abbd3bdbf93(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_18b5ae180ba95f9c6e0831819b3a5aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3005ea7f40d57fbaf1827abbd3bdbf93
    def get_inputs(self):
        return [
            paddle.uniform([171, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_fba079c5eeeb664ce2375d21f4b1c687(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[171, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_118b0d672a3bb0d68808c706c762c008(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fba079c5eeeb664ce2375d21f4b1c687
    def get_inputs(self):
        return [
            paddle.uniform([171, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0d8bad7df106ca61acc322e3618aaace(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc20a094848a67e308d963ec5296b27d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3f7d1b06be540b4a918d2acf245ad3a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cf47f1fb1d3a0c430b428dfa0bdca51
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_40c6c010d8fe72871bd214119eeaf22c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ac5b60e32655f106e5cb57a1ce901423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c6c010d8fe72871bd214119eeaf22c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_9995740a54c377c85f8688c7f37261f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9519ce64a4c630a9a2dac537c370bbba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9995740a54c377c85f8688c7f37261f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_2781c28ed490b02f28df2d6812326812(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 88, 132], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_58b77247f75fd15658585dcbeccf0bf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2781c28ed490b02f28df2d6812326812
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
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


class TestPrimitiveOp_beb1da3291be52cff8b38e91f6f0ff9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aaf2f758df5adbe4e39f7a29b963dcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_beb1da3291be52cff8b38e91f6f0ff9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aaf2f758df5adbe4e39f7a29b963dcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_beb1da3291be52cff8b38e91f6f0ff9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aaf2f758df5adbe4e39f7a29b963dcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_beb1da3291be52cff8b38e91f6f0ff9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aaf2f758df5adbe4e39f7a29b963dcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_beb1da3291be52cff8b38e91f6f0ff9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aaf2f758df5adbe4e39f7a29b963dcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_beb1da3291be52cff8b38e91f6f0ff9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aaf2f758df5adbe4e39f7a29b963dcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ba2566f4bcf666968a19b007a225745c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 60, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_473777d9c1794849702960ba886c10a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba2566f4bcf666968a19b007a225745c
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fd4d2724f21366174dcc0d4f659c95cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1a888e2572e2939c956dd8a8ab8d6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_1ecef3041998ea8cede5db2b100fb10a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_412c9f34e0dfe4236097f7b31cd94ccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ecef3041998ea8cede5db2b100fb10a
    def get_inputs(self):
        return [
            paddle.uniform([145, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_c361dd8c5b89d5539f4f008930d1832a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[145, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_787f2f6e66b52dfdd136be4b79a9e514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c361dd8c5b89d5539f4f008930d1832a
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 60], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_43d29fade117b19f40d1775eab306bfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5cc331f0a0f34974d8726d249ced0cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.42702656984329224], [-0.03797486424446106], [-0.32467591762542725], [-0.3715863823890686]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.029543638229370117], [0.02459317445755005], [-0.020875394344329834], [-0.43152952194213867]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_b40a27f9b59a57d9aa673ead924bcea6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5cc331f0a0f34974d8726d249ced0cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.016034811735153198], [-0.04461756348609924], [-0.13033828139305115], [0.1816677451133728]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.29278564453125], [-0.1465194821357727], [-0.4884662628173828], [0.46304792165756226]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_91c7cbab06edae22c4993d6d98ba0077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5cc331f0a0f34974d8726d249ced0cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3121303915977478], [-0.5012533068656921], [-0.10262379050254822], [0.3540444076061249]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_7ca20d100695f560c4730abc65bb8cd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5cc331f0a0f34974d8726d249ced0cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1762411892414093], [-0.10018911957740784], [0.4803326427936554], [-0.9249767661094666]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_f4529d30313d32c8ebc992590f88ea61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5cc331f0a0f34974d8726d249ced0cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4195559024810791], [0.258736252784729], [0.4700632691383362], [-0.007235139608383179]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2825867533683777], [-0.4766601324081421], [-0.12349918484687805], [-0.017541974782943726]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_45ef9663cbf8d7595b8c837d7ebe6f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5cc331f0a0f34974d8726d249ced0cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1602063775062561], [-0.14480668306350708], [0.40289902687072754], [-0.3244452476501465]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.38134509325027466], [0.24650567770004272], [0.34999436140060425], [-0.4619288444519043]], dtype='float32').reshape([4, 1]),
        ]


class PrimitiveOp_c120b347ffe1d4e1ded271d9f9372228(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_328c1f6c09adf2e07ef4c4b56ffbb101(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c120b347ffe1d4e1ded271d9f9372228
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.12320411205291748]], [[0.42164498567581177]], [[0.35338306427001953]], [[0.020320534706115723]], [[-0.03214597702026367]], [[-0.30578428506851196]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[-10000000000.0]], [[-10000000000.0]], [[-10000000000.0]], [[-10000000000.0]], [[-10000000000.0]], [[-10000000000.0]]], dtype='float32').reshape([6, 1, 1]),
        ]


class PrimitiveOp_7e64e8082edc6c652cf6ce9093b30699(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b86d9b1c184df6641704b240cae7b43d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e64e8082edc6c652cf6ce9093b30699
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.3774455487728119]], [[-0.14412716031074524]], [[0.09678998589515686]], [[-0.5410218238830566]], [[0.6407749056816101]], [[-0.5545647144317627]], [[-0.5502743124961853]], [[0.9246440529823303]], [[1.5379557609558105]], [[0.23808196187019348]], [[1.3766214847564697]], [[-0.6470880508422852]], [[-1.0273243188858032]], [[-0.2153625786304474]], [[-0.02013862133026123]], [[-0.3934018909931183]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_723f3308f2acc683d28af99dfab14672(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_311ed8f3713fbc52e3f347a257c82389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_723f3308f2acc683d28af99dfab14672
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_637f37fde95c5cabe8eb679aee3f3fd5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 24, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f219e9cdc41d64be80b3642f14213f89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_637f37fde95c5cabe8eb679aee3f3fd5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.8538734912872314]], [[1.3691818714141846]], [[0.3615837097167969]], [[0.2418871819972992]], [[-0.42212921380996704]], [[-0.33123114705085754]], [[1.2635080814361572]], [[-1.1388921737670898]], [[1.1786224842071533]], [[1.2062450647354126]], [[0.5758519768714905]], [[0.9751304388046265]], [[-2.3144094944000244]], [[0.8429552912712097]], [[0.33999669551849365]], [[0.8281009197235107]], [[-0.440155565738678]], [[0.8467044830322266]], [[-0.17584949731826782]], [[-0.03125643730163574]], [[0.0522325336933136]], [[0.028181128203868866]], [[-0.6039165258407593]], [[-0.08378677070140839]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_4188ede58228c460b705589b1a6124f0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 18, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4acbc45eacb0debf91d32b35ae66bb80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4188ede58228c460b705589b1a6124f0
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.362492561340332]], [[-0.22771799564361572]], [[0.2461545765399933]], [[-1.2197558879852295]], [[-1.1195178031921387]], [[0.005284219980239868]], [[1.249927282333374]], [[-0.032916873693466187]], [[-0.028332054615020752]], [[0.25710541009902954]], [[-0.5633720755577087]], [[-0.6454026103019714]], [[0.4289362132549286]], [[0.4912990927696228]], [[1.4809134006500244]], [[0.30368104577064514]], [[-0.12637946009635925]], [[0.19698841869831085]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac5b60e32655f106e5cb57a1ce901423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c6c010d8fe72871bd214119eeaf22c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2260f9cb0e378ce8bcb588c5b906179e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4188ede58228c460b705589b1a6124f0
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.4247395694255829]], [[0.844853401184082]], [[-0.23737724125385284]], [[0.24555903673171997]], [[-0.6303294897079468]], [[-1.1040548086166382]], [[0.1359490305185318]], [[-0.7146728038787842]], [[-0.6557530760765076]], [[0.3354584574699402]], [[-1.0148181915283203]], [[-0.37279313802719116]], [[0.7212413549423218]], [[0.7769100069999695]], [[-0.02441450208425522]], [[-0.38090452551841736]], [[0.6308591961860657]], [[0.5351423621177673]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac5b60e32655f106e5cb57a1ce901423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c6c010d8fe72871bd214119eeaf22c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
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


class TestPrimitiveOp_cfb7bfecc7e948556969544a1c16e565(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.32623517513275146, -0.24192336201667786, 0.27793484926223755, 0.19273710250854492, 0.4619840383529663, 0.057091474533081055], dtype='float32').reshape([6]),
            paddle.to_tensor([0.13945317268371582, 0.2671630382537842, -0.120673269033432, -0.19917404651641846, 0.26317787170410156, 0.17146217823028564], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_47d915fc8066edc8ea4490cf54feca1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.48779433965682983, 0.34922271966934204, -0.4023532271385193, 0.3019695281982422, 0.06090348958969116, 0.11269927024841309], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.21393254399299622, 0.3412080407142639, 0.2899574637413025, -0.25859230756759644, -0.26445272564888, -0.4373926520347595], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_f9a01cc2649829f1c60b7b5a90b024f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.32623517513275146, -0.24192336201667786, 0.27793484926223755, 0.19273710250854492, 0.4619840383529663, 0.057091474533081055], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.03316903114318848, 0.37095922231674194, 0.48807185888290405, 0.02101588249206543, -0.059602320194244385, 0.2165842056274414], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_1561ef43340b4af5c78e28fad126e5c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.48779433965682983, 0.34922271966934204, -0.4023532271385193, 0.3019695281982422, 0.06090348958969116, 0.11269927024841309], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.03374040126800537, -0.1420654058456421, -0.10500526428222656, -0.35081708431243896, -0.40645480155944824, -0.12897560000419617], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_b8e5651aa675bc75bbf6f4b7bb028e97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.32623517513275146, 0.2671630382537842, 0.27793484926223755, 0.19273710250854492, 0.4619840383529663, 0.17146217823028564], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.270524263381958, -0.20599150657653809, -0.06320270895957947, -0.2330784797668457, -0.28073909878730774, 0.2532276511192322], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_96d833307fea206f799e2847aa1b4fa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09afba4c000918e41517aed446418de5
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.21393254399299622, 0.34922271966934204, 0.2899574637413025, 0.3019695281982422, 0.06090348958969116, 0.11269927024841309], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.19772139191627502, 0.4174601435661316, 0.029941082000732422, -0.15820688009262085, -0.4392582178115845, -0.4687579572200775], dtype='float32').reshape([6]),
        ]


class PrimitiveOp_beb2866ad646a4731fb9007aee1a16dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_28e6c6739b1f6d3fa0d07d0e895e480f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beb2866ad646a4731fb9007aee1a16dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_40aeb0fd2b2dd8fe956835b6bdd3594d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9d5fd8fd3a119636747083813b00fddf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40aeb0fd2b2dd8fe956835b6bdd3594d
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_8354960cf4b54abc10261c4c751de7b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_06c1f428285f6308f03989e5f0ae506e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8354960cf4b54abc10261c4c751de7b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_28e2137aee26b5ebeb884e5eee400c3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 30, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5729409155775be8ae40725488d37c13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28e2137aee26b5ebeb884e5eee400c3c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.8209543228149414]], [[-0.2144690454006195]], [[-0.2849477529525757]], [[1.2590882778167725]], [[0.7037227749824524]], [[-0.6776213049888611]], [[0.5243929028511047]], [[-0.07715515792369843]], [[0.89821457862854]], [[-0.44659221172332764]], [[-0.4370743930339813]], [[0.5352632403373718]], [[-0.15174680948257446]], [[-0.9123597145080566]], [[0.36475151777267456]], [[1.467856526374817]], [[-0.2686419188976288]], [[-0.9610226154327393]], [[-0.5230721235275269]], [[-0.9574066400527954]], [[1.194096326828003]], [[-0.07337802648544312]], [[0.15819603204727173]], [[-0.1017121821641922]], [[0.4791000187397003]], [[0.9467135071754456]], [[1.7205028533935547]], [[0.576519250869751]], [[0.47908902168273926]], [[1.520843267440796]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_3d2b80435af3ce021eaadf71cf576508(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 84, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b5aff3e46702c26fb10de07249ab1d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d2b80435af3ce021eaadf71cf576508
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_c9aa775ab864c54c49b56160f6120f42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 336, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b0ad32fcd279b9d559c3c62fb633089c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9aa775ab864c54c49b56160f6120f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_d991bfc144a40c4dc77665a8845b8737(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_37fca934ffb81e15faed219aebc0f8ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d991bfc144a40c4dc77665a8845b8737
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_bcb66f66d335d8131349f351aea004f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d4c4b5cebb472b6e6c74fbdc247d046(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb66f66d335d8131349f351aea004f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7c6db7ac669a5ba6460fcf011d3b3dd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7db5615c85de2947b7aa067640bd6f20
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.3490546941757202]], [[0.1679888665676117]], [[0.2464066743850708]], [[0.5209314823150635]], [[-0.6345423460006714]], [[-0.052226483821868896]], [[-0.7103952169418335]], [[1.177135944366455]], [[-1.693554401397705]], [[-0.734485387802124]], [[0.2783372402191162]], [[0.9408055543899536]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c48945a461a948ca7b932b45da8ac59d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db9bd9300274c0e9cd7ed6222b6dde73
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_28e6c6739b1f6d3fa0d07d0e895e480f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beb2866ad646a4731fb9007aee1a16dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9d5fd8fd3a119636747083813b00fddf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40aeb0fd2b2dd8fe956835b6bdd3594d
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7b3c87b322c24101810f39d870848d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0ffdd5bf7236aed70be849b15ec5d41
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1c8fe4b5b59b3da7308318058667baf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41445b01980185f9e9f7e623d502fb75
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_9c099fb40925d421c59634e5b2938bfd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 80, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b462e3ee17e74c55582b067e13268b8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c099fb40925d421c59634e5b2938bfd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b462e3ee17e74c55582b067e13268b8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c099fb40925d421c59634e5b2938bfd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b462e3ee17e74c55582b067e13268b8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c099fb40925d421c59634e5b2938bfd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b462e3ee17e74c55582b067e13268b8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c099fb40925d421c59634e5b2938bfd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b462e3ee17e74c55582b067e13268b8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c099fb40925d421c59634e5b2938bfd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b462e3ee17e74c55582b067e13268b8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c099fb40925d421c59634e5b2938bfd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b462e3ee17e74c55582b067e13268b8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c099fb40925d421c59634e5b2938bfd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b462e3ee17e74c55582b067e13268b8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c099fb40925d421c59634e5b2938bfd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_40e0685f80793bd4c301a3522b71878d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 40, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a2a9f95786ca66bd2fd0e03f3b1d9e3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40e0685f80793bd4c301a3522b71878d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a2a9f95786ca66bd2fd0e03f3b1d9e3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40e0685f80793bd4c301a3522b71878d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a2a9f95786ca66bd2fd0e03f3b1d9e3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40e0685f80793bd4c301a3522b71878d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a2a9f95786ca66bd2fd0e03f3b1d9e3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40e0685f80793bd4c301a3522b71878d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a2a9f95786ca66bd2fd0e03f3b1d9e3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40e0685f80793bd4c301a3522b71878d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a2a9f95786ca66bd2fd0e03f3b1d9e3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40e0685f80793bd4c301a3522b71878d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a2a9f95786ca66bd2fd0e03f3b1d9e3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40e0685f80793bd4c301a3522b71878d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a2a9f95786ca66bd2fd0e03f3b1d9e3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40e0685f80793bd4c301a3522b71878d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_778c2a7e676b2bf103b99208e3436bb4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 20, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_236f9afeb17816d3ac317cdefd91ab6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_778c2a7e676b2bf103b99208e3436bb4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_236f9afeb17816d3ac317cdefd91ab6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_778c2a7e676b2bf103b99208e3436bb4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_236f9afeb17816d3ac317cdefd91ab6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_778c2a7e676b2bf103b99208e3436bb4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_236f9afeb17816d3ac317cdefd91ab6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_778c2a7e676b2bf103b99208e3436bb4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_236f9afeb17816d3ac317cdefd91ab6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_778c2a7e676b2bf103b99208e3436bb4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_236f9afeb17816d3ac317cdefd91ab6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_778c2a7e676b2bf103b99208e3436bb4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_236f9afeb17816d3ac317cdefd91ab6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_778c2a7e676b2bf103b99208e3436bb4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_236f9afeb17816d3ac317cdefd91ab6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_778c2a7e676b2bf103b99208e3436bb4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_1ca72b5c3f73e21641e1f66a07f4e6fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 10, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c7a2a3b198e1de8e9e734caf491bc5d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ca72b5c3f73e21641e1f66a07f4e6fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c7a2a3b198e1de8e9e734caf491bc5d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ca72b5c3f73e21641e1f66a07f4e6fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c7a2a3b198e1de8e9e734caf491bc5d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ca72b5c3f73e21641e1f66a07f4e6fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c7a2a3b198e1de8e9e734caf491bc5d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ca72b5c3f73e21641e1f66a07f4e6fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c7a2a3b198e1de8e9e734caf491bc5d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ca72b5c3f73e21641e1f66a07f4e6fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c7a2a3b198e1de8e9e734caf491bc5d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ca72b5c3f73e21641e1f66a07f4e6fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c7a2a3b198e1de8e9e734caf491bc5d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ca72b5c3f73e21641e1f66a07f4e6fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c7a2a3b198e1de8e9e734caf491bc5d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ca72b5c3f73e21641e1f66a07f4e6fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_936de2f7025e89407b97618a10dccfeb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 5, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d25d101112749fb53f91b8104316467a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_936de2f7025e89407b97618a10dccfeb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d25d101112749fb53f91b8104316467a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_936de2f7025e89407b97618a10dccfeb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d25d101112749fb53f91b8104316467a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_936de2f7025e89407b97618a10dccfeb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d25d101112749fb53f91b8104316467a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_936de2f7025e89407b97618a10dccfeb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d25d101112749fb53f91b8104316467a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_936de2f7025e89407b97618a10dccfeb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d25d101112749fb53f91b8104316467a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_936de2f7025e89407b97618a10dccfeb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d25d101112749fb53f91b8104316467a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_936de2f7025e89407b97618a10dccfeb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d25d101112749fb53f91b8104316467a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_936de2f7025e89407b97618a10dccfeb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_113583c9bca765fcff113c5c3182c01f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1fddfb12ce1e55d56a9acd3c9a387fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_113583c9bca765fcff113c5c3182c01f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_ccaad1c8b27d535fa8f01d6d25382881(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2efd6c58da2618552851b9580eb269ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccaad1c8b27d535fa8f01d6d25382881
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9f16307de183f24f23b857ac6332a10a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30d93ff451a143c1af4c582ec1de15bd
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d7a15b5c8cbb0593c0dd567247b1faa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_028187fcc580e99a097c9c1e40fe811e
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac5b60e32655f106e5cb57a1ce901423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c6c010d8fe72871bd214119eeaf22c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_29bb9384afefeb3a4934bf47e567936c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 20, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_89c2417c04c8859f72882cce3965614b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29bb9384afefeb3a4934bf47e567936c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_7295be19a5e047d75be3d0f66f96ca8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_511fac7318c04a6310921b8ea2a57193(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7295be19a5e047d75be3d0f66f96ca8d
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_12c5ad24f743cb81e8a0095a3dccae95(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[145, 336], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a502c3d4e4096931e916c9e99319c4a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12c5ad24f743cb81e8a0095a3dccae95
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 336], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_0e91ae64d5b069147970620069fdea92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc87839e84a9ac19e1f82b6ff576d99f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e91ae64d5b069147970620069fdea92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc87839e84a9ac19e1f82b6ff576d99f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e91ae64d5b069147970620069fdea92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc87839e84a9ac19e1f82b6ff576d99f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e91ae64d5b069147970620069fdea92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc87839e84a9ac19e1f82b6ff576d99f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e91ae64d5b069147970620069fdea92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc87839e84a9ac19e1f82b6ff576d99f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e91ae64d5b069147970620069fdea92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc87839e84a9ac19e1f82b6ff576d99f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_d7693a21a716afcbf85b862295bb50d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7f86dd05442e6612faa46448001ab97
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7693a21a716afcbf85b862295bb50d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7f86dd05442e6612faa46448001ab97
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7693a21a716afcbf85b862295bb50d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7f86dd05442e6612faa46448001ab97
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7693a21a716afcbf85b862295bb50d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7f86dd05442e6612faa46448001ab97
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7693a21a716afcbf85b862295bb50d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7f86dd05442e6612faa46448001ab97
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7693a21a716afcbf85b862295bb50d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7f86dd05442e6612faa46448001ab97
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e3d79641638b3016d46a72068daab85b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b49aa34510522763d72a6c0031adcc7
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_b0bd6b56f99ac6161f59a2ccce88c193(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39355dba4c2d85519d9ff0484f80ee94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0bd6b56f99ac6161f59a2ccce88c193
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_e7e81323087f9df3ed7e1054e1d78849(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_52494600bdc9a0bcc77328cfb676db31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7e81323087f9df3ed7e1054e1d78849
    def get_inputs(self):
        return [
            paddle.uniform([22, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_473777d9c1794849702960ba886c10a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba2566f4bcf666968a19b007a225745c
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fd4d2724f21366174dcc0d4f659c95cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1a888e2572e2939c956dd8a8ab8d6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_56d0de82d07d564ccc050fbfb6a7cd76(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 10, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5fb8978ef17961063d39824dbc8d69c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56d0de82d07d564ccc050fbfb6a7cd76
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_e10be529c065a1984139c41dbf22ed9d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f49c57ca043baf049f06413d5c70b1e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e10be529c065a1984139c41dbf22ed9d
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_155f0ed1cff2694b3dae7423a190d8d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4188ede58228c460b705589b1a6124f0
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.7528692483901978]], [[-1.7114102840423584]], [[0.7434268593788147]], [[0.2745610773563385]], [[0.5823178887367249]], [[0.3931644856929779]], [[0.10771441459655762]], [[-0.9191004037857056]], [[0.5392712950706482]], [[-0.15603142976760864]], [[-1.6454041004180908]], [[-0.326772004365921]], [[-0.5649446249008179]], [[0.756913423538208]], [[0.14601728320121765]], [[-0.5696011781692505]], [[-0.12383773922920227]], [[0.22297704219818115]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac5b60e32655f106e5cb57a1ce901423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c6c010d8fe72871bd214119eeaf22c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0d8bad7df106ca61acc322e3618aaace(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc20a094848a67e308d963ec5296b27d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3f7d1b06be540b4a918d2acf245ad3a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cf47f1fb1d3a0c430b428dfa0bdca51
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
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


class TestPrimitiveOp_4a5f445529bbb09c09983c1795e02ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6060a67461a022847a401eb2a69a853
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4a5f445529bbb09c09983c1795e02ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6060a67461a022847a401eb2a69a853
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4a5f445529bbb09c09983c1795e02ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6060a67461a022847a401eb2a69a853
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4a5f445529bbb09c09983c1795e02ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6060a67461a022847a401eb2a69a853
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4a5f445529bbb09c09983c1795e02ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6060a67461a022847a401eb2a69a853
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4a5f445529bbb09c09983c1795e02ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6060a67461a022847a401eb2a69a853
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9519ce64a4c630a9a2dac537c370bbba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9995740a54c377c85f8688c7f37261f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
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


class TestPrimitiveOp_b4c68f6ce2680ee441d5f8e4cafc1242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29df4571177cd912e25c2718ed354418
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.35990118980407715], [0.4532351493835449], [0.3253195881843567], [-0.18913787603378296], [0.22819304466247559], [0.10179847478866577]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.34053710103034973], [0.2731626033782959], [-0.39370104670524597], [0.23642903566360474], [-0.17116469144821167], [-0.068158358335495]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_4c74868b485f9e619d29e87970fad634(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29df4571177cd912e25c2718ed354418
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.36881452798843384], [0.446461021900177], [0.25204068422317505], [0.18123924732208252], [-0.04423898458480835], [0.028705358505249023]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.4686683714389801], [0.3912338614463806], [-0.45743855834007263], [-0.008816301822662354], [-0.15691471099853516], [-0.17567184567451477]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_73c447b46e069181125ea20d9b54b6ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29df4571177cd912e25c2718ed354418
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2527996897697449], [-0.1516473889350891], [-0.6138649582862854], [-0.5521718263626099], [-0.6273437142372131], [-0.37911269068717957]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_dc3fbda08ee7a3840769f760b6898bad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29df4571177cd912e25c2718ed354418
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.04148709774017334], [-0.7964882254600525], [-0.29898810386657715], [-0.2718324661254883], [-0.0077477991580963135], [-0.23272746801376343]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_cebfdf5f9d4ddd9f5462a0645a4cf7c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29df4571177cd912e25c2718ed354418
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2814381718635559], [0.3015877604484558], [-0.2885453701019287], [-0.31574276089668274], [0.2026931643486023], [-0.0327763557434082]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.10710150003433228], [0.4202950596809387], [0.20075774192810059], [-0.24555674195289612], [-0.39915066957473755], [-0.2773142158985138]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_e589344844fc7a7d7e1644fd93c7c075(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29df4571177cd912e25c2718ed354418
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.45480841398239136], [-0.3500272035598755], [-0.0469474196434021], [0.07794475555419922], [0.19924408197402954], [-0.2040221095085144]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.4103016257286072], [-0.2566860318183899], [0.030750691890716553], [-0.09059321880340576], [-0.05198678374290466], [0.11784225702285767]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_fd4d2724f21366174dcc0d4f659c95cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1a888e2572e2939c956dd8a8ab8d6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1e73e0b4e3ba0de55de2c1e175d960d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_236421ce78df60215bd2a7c878c8d700
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3a6c9212794322e7aeb6e50cdf590fe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_637f37fde95c5cabe8eb679aee3f3fd5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.004155188798904419]], [[0.08516588807106018]], [[-0.7890641093254089]], [[-0.031965479254722595]], [[1.0875771045684814]], [[2.1677112579345703]], [[-1.0167057514190674]], [[0.09207421541213989]], [[-0.5670990943908691]], [[-0.30745184421539307]], [[-0.24790889024734497]], [[0.8911083936691284]], [[1.0569405555725098]], [[0.054324671626091]], [[-0.7413526773452759]], [[0.36426860094070435]], [[1.438767910003662]], [[-1.2058196067810059]], [[1.0499999523162842]], [[0.7951397895812988]], [[-1.7341077327728271]], [[1.523306131362915]], [[0.4737389087677002]], [[1.3691688776016235]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_473777d9c1794849702960ba886c10a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba2566f4bcf666968a19b007a225745c
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fd4d2724f21366174dcc0d4f659c95cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1a888e2572e2939c956dd8a8ab8d6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b3983613168531988e0a441e36c96ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4188ede58228c460b705589b1a6124f0
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9759787321090698]], [[1.021031141281128]], [[-0.4039652943611145]], [[-0.5032721757888794]], [[0.13435417413711548]], [[0.3891170620918274]], [[0.6932864189147949]], [[1.6448830366134644]], [[-0.6507474780082703]], [[0.5266050100326538]], [[0.22864729166030884]], [[0.1677759289741516]], [[-0.30394428968429565]], [[0.23319649696350098]], [[-2.0898263454437256]], [[-0.7490589618682861]], [[0.8743141889572144]], [[0.30733129382133484]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac5b60e32655f106e5cb57a1ce901423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c6c010d8fe72871bd214119eeaf22c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_6c9986f333ff64c7e35a9670e33c0a58(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_16a441208348730a5c7fe93d9bf10fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c9986f333ff64c7e35a9670e33c0a58
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7b3c87b322c24101810f39d870848d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0ffdd5bf7236aed70be849b15ec5d41
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1c8fe4b5b59b3da7308318058667baf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41445b01980185f9e9f7e623d502fb75
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_dc1ee3c988ffeebceb71659ab5d57bab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_56e0b34c692fa5ce7cc62ceb507fe91a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc1ee3c988ffeebceb71659ab5d57bab
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9438809156417847]], [[0.4984476566314697]], [[0.1643010377883911]], [[1.003686547279358]], [[0.4528014361858368]], [[1.136164665222168]], [[-0.3540189862251282]], [[-0.12959638237953186]], [[0.311564564704895]], [[0.4708996117115021]], [[-0.2597227394580841]]]], dtype='float32').reshape([1, 11, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_622954dcb771c1e5a4bf44699528e776(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 44, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b6657cf91c94b461da2686f9bb389f2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_622954dcb771c1e5a4bf44699528e776
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c48945a461a948ca7b932b45da8ac59d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db9bd9300274c0e9cd7ed6222b6dde73
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9519ce64a4c630a9a2dac537c370bbba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9995740a54c377c85f8688c7f37261f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_94b687023f1365f8a555ab117618d184(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 14, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07995da229bf03fef67e922a75f7469c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94b687023f1365f8a555ab117618d184
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.5289448499679565]], [[-0.5488510131835938]], [[-0.7758954763412476]], [[0.6984597444534302]], [[0.06037487834692001]], [[0.5109143853187561]], [[0.3956683278083801]], [[0.576817512512207]], [[-0.7452622652053833]], [[0.9570527076721191]], [[-0.49486225843429565]], [[-0.025409698486328125]], [[-0.31995391845703125]], [[0.013590246438980103]]]], dtype='float32').reshape([1, 14, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_b222b3985c2922e29e86fce7d504551b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 56, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_287adb5cadba02410e154a6ccedb126e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b222b3985c2922e29e86fce7d504551b
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fd4d2724f21366174dcc0d4f659c95cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1a888e2572e2939c956dd8a8ab8d6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1e73e0b4e3ba0de55de2c1e175d960d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_236421ce78df60215bd2a7c878c8d700
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_d26d447c90282e8cd91c3a291dbfca91(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1ce50c90f421076c9e5091315ff8999e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d26d447c90282e8cd91c3a291dbfca91
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_5b70069c758d45982755570bd47cf526(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 336], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a1bd0f3c0f02a9e405d50384e625157(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b70069c758d45982755570bd47cf526
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_334ad820ac0827e48145498a3104787e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d4f3816a9d031b4150adaecff1a128b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_334ad820ac0827e48145498a3104787e
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2efd6c58da2618552851b9580eb269ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccaad1c8b27d535fa8f01d6d25382881
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9519ce64a4c630a9a2dac537c370bbba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9995740a54c377c85f8688c7f37261f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_bc06285743d35fbdc1304703f0e61a98(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_021c6a151be031757d51758b1e74c81c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc06285743d35fbdc1304703f0e61a98
    def get_inputs(self):
        return [
            paddle.uniform([10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_1daadf7b6f8e5e0ec8be6233f572acdd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc2c546ac5405e17f97e08d9b4fd6bf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1daadf7b6f8e5e0ec8be6233f572acdd
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b72a2cd01483ae5035b0d59feeefb30b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1bcc3d13240598ee20e5e5303d4fcc6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b72a2cd01483ae5035b0d59feeefb30b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.1408182978630066]], [[-0.6241894960403442]], [[0.6644704937934875]], [[0.45283764600753784]]]], dtype='float32').reshape([1, 4, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d24851735b36c90a534a4a38b3e36b4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e64e8082edc6c652cf6ce9093b30699
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.6145999431610107]], [[0.6093606948852539]], [[0.4309464395046234]], [[0.5707991719245911]], [[0.5129478573799133]], [[0.5181457996368408]], [[0.5002401471138]], [[0.5722014904022217]], [[0.5123381614685059]], [[0.410256028175354]], [[0.4763246178627014]], [[0.4863011837005615]], [[0.5800120830535889]], [[0.3538127541542053]], [[0.5231696963310242]], [[0.5127725601196289]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1e73e0b4e3ba0de55de2c1e175d960d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_236421ce78df60215bd2a7c878c8d700
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2efd6c58da2618552851b9580eb269ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccaad1c8b27d535fa8f01d6d25382881
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac5b60e32655f106e5cb57a1ce901423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c6c010d8fe72871bd214119eeaf22c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_16a441208348730a5c7fe93d9bf10fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c9986f333ff64c7e35a9670e33c0a58
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_5ad42ceb01a9681f80e18af1bf83da66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_01504d22893445a119cb73d69e52f5ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ad42ceb01a9681f80e18af1bf83da66
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.3233910799026489]], [[1.2262669801712036]], [[-1.0602390766143799]], [[-1.071378469467163]], [[-1.1699414253234863]], [[-0.3953385353088379]], [[-1.058081865310669]], [[0.5527987480163574]], [[-0.7256442308425903]], [[-2.049607276916504]], [[-0.04686575382947922]], [[-1.7583422660827637]], [[-0.9735700488090515]], [[-0.8473095893859863]], [[0.22320950031280518]], [[0.339469313621521]], [[0.11839225888252258]], [[-1.590423822402954]], [[-0.6902022361755371]], [[-1.5083601474761963]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_28e6c6739b1f6d3fa0d07d0e895e480f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beb2866ad646a4731fb9007aee1a16dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_5a8f16d41d4e38740c0377c85373db94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 92, 140], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f9c669e1c3af73b7437c06c852b3b17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a8f16d41d4e38740c0377c85373db94
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9f9c669e1c3af73b7437c06c852b3b17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a8f16d41d4e38740c0377c85373db94
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9f9c669e1c3af73b7437c06c852b3b17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a8f16d41d4e38740c0377c85373db94
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9f9c669e1c3af73b7437c06c852b3b17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a8f16d41d4e38740c0377c85373db94
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9f9c669e1c3af73b7437c06c852b3b17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a8f16d41d4e38740c0377c85373db94
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9f9c669e1c3af73b7437c06c852b3b17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a8f16d41d4e38740c0377c85373db94
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9f9c669e1c3af73b7437c06c852b3b17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a8f16d41d4e38740c0377c85373db94
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9f9c669e1c3af73b7437c06c852b3b17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a8f16d41d4e38740c0377c85373db94
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_a3dcc889643656a1f2d0961c65e11cce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 46, 70], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69f35c27b1a945cf82d7ae07709a6f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3dcc889643656a1f2d0961c65e11cce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_69f35c27b1a945cf82d7ae07709a6f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3dcc889643656a1f2d0961c65e11cce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_69f35c27b1a945cf82d7ae07709a6f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3dcc889643656a1f2d0961c65e11cce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_69f35c27b1a945cf82d7ae07709a6f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3dcc889643656a1f2d0961c65e11cce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_69f35c27b1a945cf82d7ae07709a6f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3dcc889643656a1f2d0961c65e11cce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_69f35c27b1a945cf82d7ae07709a6f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3dcc889643656a1f2d0961c65e11cce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_69f35c27b1a945cf82d7ae07709a6f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3dcc889643656a1f2d0961c65e11cce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_69f35c27b1a945cf82d7ae07709a6f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3dcc889643656a1f2d0961c65e11cce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_d2f1f204b22a48563474a37b4c585c87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 23, 35], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_698f21a3051ad00e51db7689506ccfa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f1f204b22a48563474a37b4c585c87
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_698f21a3051ad00e51db7689506ccfa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f1f204b22a48563474a37b4c585c87
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_698f21a3051ad00e51db7689506ccfa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f1f204b22a48563474a37b4c585c87
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_698f21a3051ad00e51db7689506ccfa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f1f204b22a48563474a37b4c585c87
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_698f21a3051ad00e51db7689506ccfa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f1f204b22a48563474a37b4c585c87
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_698f21a3051ad00e51db7689506ccfa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f1f204b22a48563474a37b4c585c87
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_698f21a3051ad00e51db7689506ccfa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f1f204b22a48563474a37b4c585c87
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_698f21a3051ad00e51db7689506ccfa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f1f204b22a48563474a37b4c585c87
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8f70172e1c6e2848fc88765e9f25221f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac1fc699d1242d3cc0e20637a40547a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8f70172e1c6e2848fc88765e9f25221f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac1fc699d1242d3cc0e20637a40547a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8f70172e1c6e2848fc88765e9f25221f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac1fc699d1242d3cc0e20637a40547a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8f70172e1c6e2848fc88765e9f25221f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac1fc699d1242d3cc0e20637a40547a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8f70172e1c6e2848fc88765e9f25221f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac1fc699d1242d3cc0e20637a40547a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8f70172e1c6e2848fc88765e9f25221f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac1fc699d1242d3cc0e20637a40547a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8f70172e1c6e2848fc88765e9f25221f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac1fc699d1242d3cc0e20637a40547a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8f70172e1c6e2848fc88765e9f25221f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac1fc699d1242d3cc0e20637a40547a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_539323fc686572d444a765c81382d5e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 6, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_06762fe6ccd9e5dd41060edd044f08b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539323fc686572d444a765c81382d5e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06762fe6ccd9e5dd41060edd044f08b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539323fc686572d444a765c81382d5e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06762fe6ccd9e5dd41060edd044f08b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539323fc686572d444a765c81382d5e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06762fe6ccd9e5dd41060edd044f08b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539323fc686572d444a765c81382d5e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06762fe6ccd9e5dd41060edd044f08b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539323fc686572d444a765c81382d5e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06762fe6ccd9e5dd41060edd044f08b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539323fc686572d444a765c81382d5e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06762fe6ccd9e5dd41060edd044f08b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539323fc686572d444a765c81382d5e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06762fe6ccd9e5dd41060edd044f08b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539323fc686572d444a765c81382d5e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_473777d9c1794849702960ba886c10a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba2566f4bcf666968a19b007a225745c
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fd4d2724f21366174dcc0d4f659c95cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1a888e2572e2939c956dd8a8ab8d6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_82f7db7102bca29b809f5bb7aadd8890(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4188ede58228c460b705589b1a6124f0
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.34709036350250244]], [[-0.4382411539554596]], [[-0.10890746116638184]], [[0.9488625526428223]], [[-0.3571561574935913]], [[0.38370391726493835]], [[0.6414154171943665]], [[-0.7954689264297485]], [[0.5097373723983765]], [[-0.028549641370773315]], [[-0.094255730509758]], [[0.7197317481040955]], [[-0.2976454496383667]], [[-1.2454389333724976]], [[0.39047279953956604]], [[-0.9860487580299377]], [[-0.1868377923965454]], [[-0.7577764987945557]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac5b60e32655f106e5cb57a1ce901423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c6c010d8fe72871bd214119eeaf22c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
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


class TestPrimitiveOp_1cbcedb3ba854264255d7ff37ed30d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de944bb40bb74a45d9d908ccb174da25
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1cbcedb3ba854264255d7ff37ed30d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de944bb40bb74a45d9d908ccb174da25
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1cbcedb3ba854264255d7ff37ed30d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de944bb40bb74a45d9d908ccb174da25
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1cbcedb3ba854264255d7ff37ed30d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de944bb40bb74a45d9d908ccb174da25
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1cbcedb3ba854264255d7ff37ed30d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de944bb40bb74a45d9d908ccb174da25
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1cbcedb3ba854264255d7ff37ed30d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de944bb40bb74a45d9d908ccb174da25
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2efd6c58da2618552851b9580eb269ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccaad1c8b27d535fa8f01d6d25382881
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_71ad16da1ed88d83e7befb7ce7d93fa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_637f37fde95c5cabe8eb679aee3f3fd5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.01858934760093689]], [[-1.5713950395584106]], [[-0.4968070387840271]], [[-0.5363447666168213]], [[-0.06203043460845947]], [[0.3650085926055908]], [[0.6366583108901978]], [[0.44355010986328125]], [[0.24651259183883667]], [[0.3250676393508911]], [[0.4283791780471802]], [[-1.5106699466705322]], [[0.07251378893852234]], [[-1.4423565864562988]], [[-0.6426616907119751]], [[0.39343732595443726]], [[-0.007804661989212036]], [[0.07563605904579163]], [[-1.0879325866699219]], [[-0.6899603009223938]], [[0.12446397542953491]], [[1.5286474227905273]], [[1.1897865533828735]], [[1.2460485696792603]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_70cb339b3f7a24cedc26cc1de6728434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aaf550957e26b3fa1ff256401484b75d
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_5d9380d9efce211658d4e7c1eadb0dbf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_77f9aa8ffe79fe255d687138567e87d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d9380d9efce211658d4e7c1eadb0dbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_088203e1ee31d971c0ea6f9ad3ceedb2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 960], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a279aa40c5197642373a5d9e11d23f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_088203e1ee31d971c0ea6f9ad3ceedb2
    def get_inputs(self):
        return [
            paddle.uniform([1, 960], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_1ac5217bef83b547fe873b9b1a5f4400(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76211934d2a84fd2ee5cf8ea4fc0ccfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1ac5217bef83b547fe873b9b1a5f4400(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76211934d2a84fd2ee5cf8ea4fc0ccfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1ac5217bef83b547fe873b9b1a5f4400(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76211934d2a84fd2ee5cf8ea4fc0ccfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1ac5217bef83b547fe873b9b1a5f4400(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76211934d2a84fd2ee5cf8ea4fc0ccfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1ac5217bef83b547fe873b9b1a5f4400(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76211934d2a84fd2ee5cf8ea4fc0ccfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1ac5217bef83b547fe873b9b1a5f4400(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76211934d2a84fd2ee5cf8ea4fc0ccfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b5aff3e46702c26fb10de07249ab1d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d2b80435af3ce021eaadf71cf576508
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b0ad32fcd279b9d559c3c62fb633089c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9aa775ab864c54c49b56160f6120f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c839318a0d641aaf9cb6e81e5e29bd3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c839318a0d641aaf9cb6e81e5e29bd3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a15b41906019eb0139cbc79eddc83ae4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a15b41906019eb0139cbc79eddc83ae4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c839318a0d641aaf9cb6e81e5e29bd3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c839318a0d641aaf9cb6e81e5e29bd3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f72c3d115bb396ead2a56a90ac707fa6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4116, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7be4f56484f9e0119320736a4ce7ac1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f72c3d115bb396ead2a56a90ac707fa6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0484c9d013b12a30430fcd0c3e022673(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e4164e3d60b6586339e17ebf8018de2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0484c9d013b12a30430fcd0c3e022673
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
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


class TestPrimitiveOp_d17cdfddf793e6d9c3bede675b97d546(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f60b8e268f203712cdfd228e8f3ad2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d17cdfddf793e6d9c3bede675b97d546(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f60b8e268f203712cdfd228e8f3ad2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d17cdfddf793e6d9c3bede675b97d546(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f60b8e268f203712cdfd228e8f3ad2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d17cdfddf793e6d9c3bede675b97d546(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f60b8e268f203712cdfd228e8f3ad2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d17cdfddf793e6d9c3bede675b97d546(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f60b8e268f203712cdfd228e8f3ad2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d17cdfddf793e6d9c3bede675b97d546(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f60b8e268f203712cdfd228e8f3ad2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_085b1a50413c7021b733e0a5979d6839(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 156], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80e6cefa213c78ab3b710edf5cdcf64f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_085b1a50413c7021b733e0a5979d6839
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_2c091436de918a6eb83e06e32c48914a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 624], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 624], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c7d21241e5c786c78ec9436f83c1633e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c091436de918a6eb83e06e32c48914a
    def get_inputs(self):
        return [
            paddle.uniform([1, 624], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 624], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7b3c87b322c24101810f39d870848d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0ffdd5bf7236aed70be849b15ec5d41
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1c8fe4b5b59b3da7308318058667baf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41445b01980185f9e9f7e623d502fb75
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_c91c323953884e32403c8db8e7f53be2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 46, 70], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a6e1fb6e25e6fef92491e1c9a04e782(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c91c323953884e32403c8db8e7f53be2
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_971f74dacec3cc9eb0785c20efb1c69e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 25, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f76aab582b46dc7fc5579bd6f60caace(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_971f74dacec3cc9eb0785c20efb1c69e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.5991321802139282]], [[-1.4896169900894165]], [[0.4777927100658417]], [[-0.16868990659713745]], [[-0.6486026644706726]], [[-0.5278662443161011]], [[-1.0345277786254883]], [[1.5711188316345215]], [[0.1935962736606598]], [[-0.9337688684463501]], [[-0.19633695483207703]], [[1.2224235534667969]], [[0.21996624767780304]], [[0.6351614594459534]], [[-0.3325824737548828]], [[0.20418868958950043]], [[-0.1274154931306839]], [[-0.2354564517736435]], [[0.03334328532218933]], [[0.720421314239502]], [[0.457146555185318]], [[-0.6815547347068787]], [[0.151421457529068]], [[0.8483597040176392]], [[-0.4560200572013855]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9f16307de183f24f23b857ac6332a10a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30d93ff451a143c1af4c582ec1de15bd
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_70cb339b3f7a24cedc26cc1de6728434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aaf550957e26b3fa1ff256401484b75d
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_08289d443e6f5b0e48d90155c48b479b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 50, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0153f0324fbe360bb910fdb118a7395e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08289d443e6f5b0e48d90155c48b479b
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_467c4596305cfa8adfbad479de79a399(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 200, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_629724ccc231d72e797a61c9aeefb94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_467c4596305cfa8adfbad479de79a399
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_f3caf76c83efad3061bdd0de58fb0eb4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_27043036fde83f29a1a477f9874591c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3caf76c83efad3061bdd0de58fb0eb4
    def get_inputs(self):
        return [
            paddle.uniform([171, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_22384602c5c6140198540c761c556928(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[171, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_26221d0e94f43e4e7c0af473c01c6bf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22384602c5c6140198540c761c556928
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aabb456b250bdaadc2692d0f68f65f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ad42ceb01a9681f80e18af1bf83da66
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.8830095529556274]], [[-0.4852551221847534]], [[-0.6338146328926086]], [[1.0516648292541504]], [[-1.284122109413147]], [[-0.663877546787262]], [[0.22535234689712524]], [[0.23741841316223145]], [[-0.5678741931915283]], [[0.5370608568191528]], [[1.0816140174865723]], [[-0.9392238259315491]], [[-1.162736415863037]], [[-0.24943214654922485]], [[-0.5026562809944153]], [[1.0017340183258057]], [[0.44102561473846436]], [[-2.2065985202789307]], [[0.6268479228019714]], [[-0.18834733963012695]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_28e6c6739b1f6d3fa0d07d0e895e480f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beb2866ad646a4731fb9007aee1a16dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_8995c1960637d984e7425d137883184f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d5ea12fe85286bc9000b8a326a4073e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8995c1960637d984e7425d137883184f
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_10c7e4c433260f80f027b5feeca16067(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 336], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2adc46d7c8f0906755f56f4186fd341e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10c7e4c433260f80f027b5feeca16067
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 336], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_c34b32b1ea66605279a2ac0b74514239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b5805ec7af0d88bfb6e9accc93507e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c34b32b1ea66605279a2ac0b74514239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b5805ec7af0d88bfb6e9accc93507e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c34b32b1ea66605279a2ac0b74514239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b5805ec7af0d88bfb6e9accc93507e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c34b32b1ea66605279a2ac0b74514239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b5805ec7af0d88bfb6e9accc93507e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c34b32b1ea66605279a2ac0b74514239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b5805ec7af0d88bfb6e9accc93507e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c34b32b1ea66605279a2ac0b74514239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b5805ec7af0d88bfb6e9accc93507e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_202836fe2fc0d878e1509704cd053a7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7db5615c85de2947b7aa067640bd6f20
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.6403800845146179]], [[0.1651822328567505]], [[0.08752718567848206]], [[-0.9447072744369507]], [[0.00709986686706543]], [[-0.16313108801841736]], [[-1.3288387060165405]], [[-0.165279358625412]], [[0.04951098561286926]], [[-0.7705563902854919]], [[-0.0449255108833313]], [[-0.7309733629226685]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c48945a461a948ca7b932b45da8ac59d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db9bd9300274c0e9cd7ed6222b6dde73
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_63663d4e077d0fd2af7204edab4be4e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_accdc483fdf123138ea8105ac645b9ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63663d4e077d0fd2af7204edab4be4e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_1a7fd6186632427fa3083e154ee401bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b8aee6b680b5c3367f9e5c7f8928d033(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7fd6186632427fa3083e154ee401bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_caf03fffdd593d96ac55d69fc4971003(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 14, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5745c67a525d4d30947c5346975fa578(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_caf03fffdd593d96ac55d69fc4971003
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_f51a514d025d05494e3cd6ab06ffe762(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_81258d9fa697da174e09674d93834b76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f51a514d025d05494e3cd6ab06ffe762
    def get_inputs(self):
        return [
            paddle.uniform([145, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_66158e44cddb85231411ec2957c58914(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[145, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f589f8cb1c0c360fced839181efc2a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66158e44cddb85231411ec2957c58914
    def get_inputs(self):
        return [
            paddle.uniform([145, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 36], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_a923962b0ef84fad1bd205b706f93587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec36b70774ed7c07b274b3f151805f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a923962b0ef84fad1bd205b706f93587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec36b70774ed7c07b274b3f151805f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a923962b0ef84fad1bd205b706f93587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec36b70774ed7c07b274b3f151805f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a923962b0ef84fad1bd205b706f93587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec36b70774ed7c07b274b3f151805f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a923962b0ef84fad1bd205b706f93587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec36b70774ed7c07b274b3f151805f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a923962b0ef84fad1bd205b706f93587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec36b70774ed7c07b274b3f151805f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0d5ea12fe85286bc9000b8a326a4073e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8995c1960637d984e7425d137883184f
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2adc46d7c8f0906755f56f4186fd341e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10c7e4c433260f80f027b5feeca16067
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_1e01204ad51cf8504e27f9f6b7a0b524(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[390, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_68dd8aca2405fdd015e9411a78e57865(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e01204ad51cf8504e27f9f6b7a0b524
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_68dd8aca2405fdd015e9411a78e57865(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e01204ad51cf8504e27f9f6b7a0b524
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_ca107667fe728b7e881947d6ae9f1f17(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 14, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_972effd29a2e176c868803e5fd28a4e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca107667fe728b7e881947d6ae9f1f17
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
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


class TestPrimitiveOp_56c3a20c7d2769942f432e8470c4af85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23bd35d07a8597402908f29ca134acd7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11598128080368042], [-0.19975826144218445], [-0.01109778881072998], [0.1941872239112854], [-0.17401427030563354]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.30723488330841064], [0.35330432653427124], [-0.4845125675201416], [-0.1398959755897522], [-0.31891727447509766]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_37e18fc7c4c984c634a68f57026b6868(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23bd35d07a8597402908f29ca134acd7
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.49989163875579834], [-0.10271179676055908], [0.22155678272247314], [-0.12957623600959778], [0.27495449781417847]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.24333494901657104], [0.034552812576293945], [0.33392423391342163], [-0.12716543674468994], [0.4143192768096924]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_4c368ee33effd8aee366c2de10c39fac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23bd35d07a8597402908f29ca134acd7
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.6303538680076599], [-0.7695514559745789], [-0.234910786151886], [-0.6752417087554932], [0.5591869354248047]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_733f6f2c876fafa199e10c0f4e18c3b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23bd35d07a8597402908f29ca134acd7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.13120320439338684], [0.1371973752975464], [-0.3553307056427002], [-0.1541731357574463], [-0.47886979579925537]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_e2d74ece33ce72c9c12cd4be22072aa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23bd35d07a8597402908f29ca134acd7
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2732882499694824], [0.21844327449798584], [0.1365358829498291], [0.1824830174446106], [0.38517266511917114]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.32311898469924927], [-0.4162471294403076], [-0.24600857496261597], [-0.4810544550418854], [0.46119225025177]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_bb708d2fa49b442261449a63c11eb99d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23bd35d07a8597402908f29ca134acd7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.47203218936920166], [0.33248358964920044], [-0.021406471729278564], [-0.28133857250213623], [0.33753883838653564]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.1121317446231842], [0.17175018787384033], [0.2869950532913208], [-0.10179561376571655], [-0.06455051898956299]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_0d8bad7df106ca61acc322e3618aaace(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc20a094848a67e308d963ec5296b27d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3f7d1b06be540b4a918d2acf245ad3a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cf47f1fb1d3a0c430b428dfa0bdca51
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_be160504a558d70daa55c22dc7f4806a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be160504a558d70daa55c22dc7f4806a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4b50f6a55b9ed6664fcd25273c7b6958(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4b50f6a55b9ed6664fcd25273c7b6958(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be160504a558d70daa55c22dc7f4806a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be160504a558d70daa55c22dc7f4806a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d132eacb1e126448f2ecf9489c59ec1e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3549, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b172a8f3c86e3e3e2baffe7f7b915644(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d132eacb1e126448f2ecf9489c59ec1e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9519ce64a4c630a9a2dac537c370bbba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9995740a54c377c85f8688c7f37261f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_b51b768f79e20c6185bf4d3300c313cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 11, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ffaea8716f6983496a24d88970cbd9e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b51b768f79e20c6185bf4d3300c313cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_97ad36a27c2e642436ff8e15a33cd1a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f02d6f992f2c32123b4b2550b2f65fe9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97ad36a27c2e642436ff8e15a33cd1a8
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.8249450922012329]], [[0.0974121168255806]], [[0.41337549686431885]], [[0.2480238974094391]], [[0.38506215810775757]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_411bb89e19b1a9d716ca821e15c913b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 10, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70bc15777800c833bc0efff592750d56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_411bb89e19b1a9d716ca821e15c913b8
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.3922470808029175]], [[0.5626466274261475]], [[-0.2878000736236572]], [[-0.12519603967666626]], [[1.5134825706481934]], [[0.33570337295532227]], [[0.4328838884830475]], [[-0.19557306170463562]], [[-0.8901615142822266]], [[-0.3975234031677246]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b3cd47aad2de49071f6eafb418cc053a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ad42ceb01a9681f80e18af1bf83da66
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.31563007831573486]], [[0.035672158002853394]], [[-1.2013294696807861]], [[-0.7649065256118774]], [[-1.1189337968826294]], [[0.20217695832252502]], [[0.20550110936164856]], [[-0.9239870309829712]], [[-0.9736810922622681]], [[0.3525931239128113]], [[0.38546642661094666]], [[0.33284685015678406]], [[-0.064310222864151]], [[0.6999605894088745]], [[-0.10375915467739105]], [[0.15202581882476807]], [[0.44090893864631653]], [[1.2819464206695557]], [[-0.10602270066738129]], [[0.11653076112270355]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_d2d0831c32986a72b16572f8d4eaa1a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 96, 144], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_db8ddacdf0dcbb38720a7bb2d8000b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2d0831c32986a72b16572f8d4eaa1a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_db8ddacdf0dcbb38720a7bb2d8000b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2d0831c32986a72b16572f8d4eaa1a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_db8ddacdf0dcbb38720a7bb2d8000b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2d0831c32986a72b16572f8d4eaa1a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_db8ddacdf0dcbb38720a7bb2d8000b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2d0831c32986a72b16572f8d4eaa1a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_db8ddacdf0dcbb38720a7bb2d8000b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2d0831c32986a72b16572f8d4eaa1a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_db8ddacdf0dcbb38720a7bb2d8000b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2d0831c32986a72b16572f8d4eaa1a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_db8ddacdf0dcbb38720a7bb2d8000b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2d0831c32986a72b16572f8d4eaa1a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_db8ddacdf0dcbb38720a7bb2d8000b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2d0831c32986a72b16572f8d4eaa1a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_e2882eb1014f604daddbec5559eda7a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 48, 72], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_89c516c7d1e449b0d1cc20a79055fd1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2882eb1014f604daddbec5559eda7a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_89c516c7d1e449b0d1cc20a79055fd1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2882eb1014f604daddbec5559eda7a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_89c516c7d1e449b0d1cc20a79055fd1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2882eb1014f604daddbec5559eda7a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_89c516c7d1e449b0d1cc20a79055fd1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2882eb1014f604daddbec5559eda7a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_89c516c7d1e449b0d1cc20a79055fd1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2882eb1014f604daddbec5559eda7a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_89c516c7d1e449b0d1cc20a79055fd1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2882eb1014f604daddbec5559eda7a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_89c516c7d1e449b0d1cc20a79055fd1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2882eb1014f604daddbec5559eda7a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_89c516c7d1e449b0d1cc20a79055fd1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2882eb1014f604daddbec5559eda7a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_2ed70527ddcc6f59267645b4e1e1de1f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 24, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_76111dc6fa1afb490065651bcf59233d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed70527ddcc6f59267645b4e1e1de1f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_76111dc6fa1afb490065651bcf59233d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed70527ddcc6f59267645b4e1e1de1f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_76111dc6fa1afb490065651bcf59233d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed70527ddcc6f59267645b4e1e1de1f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_76111dc6fa1afb490065651bcf59233d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed70527ddcc6f59267645b4e1e1de1f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_76111dc6fa1afb490065651bcf59233d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed70527ddcc6f59267645b4e1e1de1f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_76111dc6fa1afb490065651bcf59233d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed70527ddcc6f59267645b4e1e1de1f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_76111dc6fa1afb490065651bcf59233d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed70527ddcc6f59267645b4e1e1de1f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_76111dc6fa1afb490065651bcf59233d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed70527ddcc6f59267645b4e1e1de1f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8f70172e1c6e2848fc88765e9f25221f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac1fc699d1242d3cc0e20637a40547a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8f70172e1c6e2848fc88765e9f25221f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac1fc699d1242d3cc0e20637a40547a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8f70172e1c6e2848fc88765e9f25221f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac1fc699d1242d3cc0e20637a40547a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8f70172e1c6e2848fc88765e9f25221f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac1fc699d1242d3cc0e20637a40547a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8f70172e1c6e2848fc88765e9f25221f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac1fc699d1242d3cc0e20637a40547a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8f70172e1c6e2848fc88765e9f25221f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac1fc699d1242d3cc0e20637a40547a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8f70172e1c6e2848fc88765e9f25221f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac1fc699d1242d3cc0e20637a40547a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8f70172e1c6e2848fc88765e9f25221f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac1fc699d1242d3cc0e20637a40547a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06762fe6ccd9e5dd41060edd044f08b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539323fc686572d444a765c81382d5e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06762fe6ccd9e5dd41060edd044f08b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539323fc686572d444a765c81382d5e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06762fe6ccd9e5dd41060edd044f08b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539323fc686572d444a765c81382d5e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06762fe6ccd9e5dd41060edd044f08b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539323fc686572d444a765c81382d5e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06762fe6ccd9e5dd41060edd044f08b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539323fc686572d444a765c81382d5e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06762fe6ccd9e5dd41060edd044f08b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539323fc686572d444a765c81382d5e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06762fe6ccd9e5dd41060edd044f08b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539323fc686572d444a765c81382d5e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06762fe6ccd9e5dd41060edd044f08b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539323fc686572d444a765c81382d5e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7b3c87b322c24101810f39d870848d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0ffdd5bf7236aed70be849b15ec5d41
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1c8fe4b5b59b3da7308318058667baf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41445b01980185f9e9f7e623d502fb75
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_77b1c1597a2df270d11894e0df6e58eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 18], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bcc5f7f279daf4e62a928d0c41df2211(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77b1c1597a2df270d11894e0df6e58eb
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0396038293838501, -0.5936009883880615, -0.8368020057678223, 0.9012128114700317, -0.8783658742904663, -0.060722291469573975, -0.13587230443954468, -0.6630713939666748, -1.3531869649887085, -0.29105275869369507, 0.29511356353759766, -0.24574829638004303, 0.5816411972045898, -0.8826116919517517, 1.4558321237564087, -0.17313632369041443, -1.1122336387634277, -0.14679209887981415]], dtype='float32').reshape([1, 18]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_12302ca0802ba1b3aaa1f76764315966(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc4dee981964353e353a9def1f5fc712(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12302ca0802ba1b3aaa1f76764315966
    def get_inputs(self):
        return [
            paddle.uniform([1, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a313482146999dae5f3ceab64d117e1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4188ede58228c460b705589b1a6124f0
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.707927942276001]], [[1.2280206680297852]], [[-0.2430729866027832]], [[-0.660903811454773]], [[0.07873739302158356]], [[0.03257882595062256]], [[-0.14508819580078125]], [[-0.15961822867393494]], [[-0.8023476004600525]], [[0.33743155002593994]], [[-0.7300020456314087]], [[-0.050976455211639404]], [[0.93083655834198]], [[0.2144901603460312]], [[0.2168939858675003]], [[0.5088299512863159]], [[-0.48700278997421265]], [[0.20958083868026733]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac5b60e32655f106e5cb57a1ce901423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c6c010d8fe72871bd214119eeaf22c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_ee0559346af043b8b0bb5b8097da2b89(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 44, 66], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e6d856643f81dc832de0d11c9bb10c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee0559346af043b8b0bb5b8097da2b89
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2efd6c58da2618552851b9580eb269ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccaad1c8b27d535fa8f01d6d25382881
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c6612c5fb3dfa2aceabb60ed1f63bfc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c6612c5fb3dfa2aceabb60ed1f63bfc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e87d5fac1afd538d8e044239f6f02746(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e87d5fac1afd538d8e044239f6f02746(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c6612c5fb3dfa2aceabb60ed1f63bfc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c6612c5fb3dfa2aceabb60ed1f63bfc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5757f6d6fa2202fa34e59f2ee6b0c3af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6069, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6e12278f2b9a856a58fece2ce9b0a9a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5757f6d6fa2202fa34e59f2ee6b0c3af
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6069, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fd4d2724f21366174dcc0d4f659c95cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1a888e2572e2939c956dd8a8ab8d6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
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


class TestPrimitiveOp_7e43d3bd20f5359df168f79c87c6c90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b16e12eb901d5e13c56f334912243682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e43d3bd20f5359df168f79c87c6c90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b16e12eb901d5e13c56f334912243682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e43d3bd20f5359df168f79c87c6c90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b16e12eb901d5e13c56f334912243682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e43d3bd20f5359df168f79c87c6c90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b16e12eb901d5e13c56f334912243682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e43d3bd20f5359df168f79c87c6c90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b16e12eb901d5e13c56f334912243682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e43d3bd20f5359df168f79c87c6c90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b16e12eb901d5e13c56f334912243682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c050acd2bc509667af091a1562c32327(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2bf32ae429449077df2c15f036cab7cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c050acd2bc509667af091a1562c32327
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_4d18fcbeff94d8a1d3cdd4643352cbc6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[171, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b6154d213466e00013916e3314bb5e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d18fcbeff94d8a1d3cdd4643352cbc6
    def get_inputs(self):
        return [
            paddle.uniform([171, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9519ce64a4c630a9a2dac537c370bbba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9995740a54c377c85f8688c7f37261f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_4949d9adbe9945f1369cfd6c417daba3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fb0a29410e9d8399a0651513ee06763d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4949d9adbe9945f1369cfd6c417daba3
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_8397557ee9d1ef73a1bff98a59e03643(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_15149a4d4eaa1728745af577a7138d02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8397557ee9d1ef73a1bff98a59e03643
    def get_inputs(self):
        return [
            paddle.uniform([10, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7af53569cf6eb3e207d2d6b73a2fbce9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b18e13f08ec7601bd78d0fb61bbfbad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7af53569cf6eb3e207d2d6b73a2fbce9
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bcf27214249be6798d0d1f4c8ba8e5c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28e2137aee26b5ebeb884e5eee400c3c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.157795250415802]], [[1.0834922790527344]], [[0.2573573589324951]], [[-0.3707621693611145]], [[-0.4482473134994507]], [[0.2863411605358124]], [[-0.15463006496429443]], [[-0.10797445476055145]], [[1.1713635921478271]], [[-1.4052228927612305]], [[-0.6565923690795898]], [[0.6188511848449707]], [[-1.034160852432251]], [[0.9432393312454224]], [[-0.9185549020767212]], [[1.8663455247879028]], [[1.7539775371551514]], [[-0.06830647587776184]], [[1.1453907489776611]], [[-1.0775909423828125]], [[1.1128240823745728]], [[-1.4711804389953613]], [[-0.630695641040802]], [[0.5184199810028076]], [[-2.2997078895568848]], [[0.10955497622489929]], [[-0.4032912254333496]], [[-2.232987403869629]], [[-0.972823977470398]], [[0.9740720987319946]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
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


class TestPrimitiveOp_75f6f23d83441d434cc7b257f08bff02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_770bf8419b3ec186116d5948a10b0232
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4368530511856079], [0.46684616804122925], [0.23317939043045044], [0.343126118183136], [0.37048572301864624], [0.12745672464370728], [-0.2904985547065735], [0.3288099765777588], [0.3160330057144165]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.2951025366783142], [0.1575915813446045], [0.18225038051605225], [0.010741472244262695], [0.4215221405029297], [0.45149946212768555], [-0.3987027704715729], [-0.3669936954975128], [0.3321917653083801]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_94b7d7627d226a96590cf0f172e0c9c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_770bf8419b3ec186116d5948a10b0232
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.3653436005115509], [-0.4201619625091553], [-0.20712703466415405], [0.1771603226661682], [-0.06482309103012085], [0.40892404317855835], [0.18776559829711914], [-0.0013248622417449951], [-0.025652199983596802]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.10920676589012146], [0.2152063250541687], [0.19116997718811035], [0.02454829216003418], [0.1994050145149231], [-0.48527729511260986], [0.4096217751502991], [-0.12472957372665405], [-0.4282859265804291]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_97c929777f609d77409a330d37bae6b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_770bf8419b3ec186116d5948a10b0232
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.17345154285430908], [-0.4735841453075409], [-0.5550603866577148], [-0.4251040518283844], [-0.9014554023742676], [-0.536878228187561], [0.2378315031528473], [-0.77287757396698], [-0.7646393775939941]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_817c2f384ea388d4f40a719e71838933(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_770bf8419b3ec186116d5948a10b0232
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1000869870185852], [0.10020291805267334], [-0.45769524574279785], [-0.30565184354782104], [-0.31859248876571655], [-0.8693258166313171], [-0.8254417181015015], [-0.48135021328926086], [-0.3163154125213623]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_46dd9a4ab266ddbe8a8688f7990ef744(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_770bf8419b3ec186116d5948a10b0232
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.26340150833129883], [0.46034830808639526], [0.16207528114318848], [-0.08197793364524841], [-0.4799332618713379], [-0.08537876605987549], [0.2270095944404602], [-0.21020689606666565], [-0.432447612285614]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.30383044481277466], [-0.0067379772663116455], [-0.3218809962272644], [0.3049466609954834], [0.342303991317749], [0.4855678081512451], [-0.052667051553726196], [-0.4440675973892212], [0.3661329746246338]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_5abb799168dcb8595636764b243b05ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_770bf8419b3ec186116d5948a10b0232
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07887744903564453], [0.31540924310684204], [-0.2665252685546875], [-0.12849152088165283], [0.4635170102119446], [-0.1288277506828308], [-0.4158199727535248], [0.08123522996902466], [-0.3419676125049591]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.009119778871536255], [0.35780632495880127], [-0.07009202241897583], [0.13681113719940186], [-0.11918747425079346], [-0.4604017734527588], [0.17320948839187622], [-0.48267507553100586], [-0.1678488850593567]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_05cb5e2dd78ced3ffaa2c24f146675b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_971f74dacec3cc9eb0785c20efb1c69e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.40766197443008423]], [[1.0775163173675537]], [[1.0531301498413086]], [[1.7806501388549805]], [[-0.5709245800971985]], [[0.5911766290664673]], [[-1.524393916130066]], [[1.8195273876190186]], [[-0.1702926754951477]], [[-1.6537835597991943]], [[-0.04967677593231201]], [[0.5408480167388916]], [[0.911496102809906]], [[0.8087146878242493]], [[-1.533637285232544]], [[-0.2883208394050598]], [[-0.6318744421005249]], [[-0.5263152122497559]], [[0.24096447229385376]], [[0.3817829489707947]], [[-0.927861213684082]], [[-0.3428639769554138]], [[-0.19237016141414642]], [[-2.1798369884490967]], [[0.32453083992004395]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9f16307de183f24f23b857ac6332a10a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30d93ff451a143c1af4c582ec1de15bd
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7b18e13f08ec7601bd78d0fb61bbfbad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7af53569cf6eb3e207d2d6b73a2fbce9
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2efd6c58da2618552851b9580eb269ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccaad1c8b27d535fa8f01d6d25382881
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37fca934ffb81e15faed219aebc0f8ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d991bfc144a40c4dc77665a8845b8737
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d4c4b5cebb472b6e6c74fbdc247d046(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb66f66d335d8131349f351aea004f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a9db9503b62600abcdba246528bb1aa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28e2137aee26b5ebeb884e5eee400c3c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.659741997718811]], [[-0.14252042770385742]], [[0.5310404300689697]], [[-1.0141054391860962]], [[-1.245516300201416]], [[0.20977240800857544]], [[1.5703617334365845]], [[-0.920410692691803]], [[0.35564786195755005]], [[-0.8567886352539062]], [[0.7299639582633972]], [[0.37659987807273865]], [[-0.938434898853302]], [[-0.11922469735145569]], [[-1.8674848079681396]], [[-0.7353026866912842]], [[0.5799760222434998]], [[0.1581488847732544]], [[-0.17702072858810425]], [[0.5483694076538086]], [[0.9400527477264404]], [[-0.39403751492500305]], [[-1.253521203994751]], [[-0.5984179377555847]], [[0.8154575824737549]], [[0.030748486518859863]], [[-0.623201847076416]], [[-1.633205533027649]], [[0.09313669800758362]], [[-0.3143114447593689]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cf6b48ac690e7116e4d8c73ff465681d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c120b347ffe1d4e1ded271d9f9372228
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.44863361120224]], [[-0.4261649250984192]], [[-0.19829675555229187]], [[0.01481938362121582]], [[0.43206214904785156]], [[0.2988715171813965]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[-10000000000.0]], [[-10000000000.0]], [[-10000000000.0]], [[-10000000000.0]], [[-10000000000.0]], [[-10000000000.0]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_7b3c87b322c24101810f39d870848d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0ffdd5bf7236aed70be849b15ec5d41
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1c8fe4b5b59b3da7308318058667baf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41445b01980185f9e9f7e623d502fb75
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_16a441208348730a5c7fe93d9bf10fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c9986f333ff64c7e35a9670e33c0a58
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c48945a461a948ca7b932b45da8ac59d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db9bd9300274c0e9cd7ed6222b6dde73
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4a431d1b2a54870647ded70e565cf9af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ad42ceb01a9681f80e18af1bf83da66
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-1.1088380813598633]], [[0.4604218602180481]], [[1.2884106636047363]], [[0.061867326498031616]], [[1.200793743133545]], [[-0.9213391542434692]], [[-0.4149431586265564]], [[-0.1367504894733429]], [[-0.5032020807266235]], [[1.3184396028518677]], [[0.7461247444152832]], [[0.4268416166305542]], [[-0.37588995695114136]], [[-0.2647217810153961]], [[-0.35490959882736206]], [[0.910429060459137]], [[-0.050569891929626465]], [[0.7715167999267578]], [[0.18232092261314392]], [[-0.7155232429504395]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_28e6c6739b1f6d3fa0d07d0e895e480f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beb2866ad646a4731fb9007aee1a16dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_1dadb73533dbffdcc67065c76aab1e71(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_476ca2df30c7366e470df046db9843e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1dadb73533dbffdcc67065c76aab1e71
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d564d93dcbbf1fa1b5543a2cd00b4b30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d564d93dcbbf1fa1b5543a2cd00b4b30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5179f5a532c1f2a0c97cda1dd22dd769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5179f5a532c1f2a0c97cda1dd22dd769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d564d93dcbbf1fa1b5543a2cd00b4b30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d564d93dcbbf1fa1b5543a2cd00b4b30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7be4f56484f9e0119320736a4ce7ac1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f72c3d115bb396ead2a56a90ac707fa6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a17b1499b850a96441538751b16a3b04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1bfcb9d96c30936d8e867ade4d409319(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a17b1499b850a96441538751b16a3b04
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0153f0324fbe360bb910fdb118a7395e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08289d443e6f5b0e48d90155c48b479b
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_629724ccc231d72e797a61c9aeefb94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_467c4596305cfa8adfbad479de79a399
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_dfa50db3a210a924b207f42a18d63bc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ad42ceb01a9681f80e18af1bf83da66
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.4741549789905548]], [[-0.21687832474708557]], [[-0.16935011744499207]], [[-1.6298341751098633]], [[0.9363090991973877]], [[-1.16203773021698]], [[0.30306166410446167]], [[0.13606181740760803]], [[0.8438928127288818]], [[1.3988460302352905]], [[-0.7327039837837219]], [[-1.1832661628723145]], [[0.521054208278656]], [[0.09980827569961548]], [[-1.193886399269104]], [[-0.5143295526504517]], [[0.3014901876449585]], [[-0.23603451251983643]], [[0.3293527662754059]], [[0.8703634738922119]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_28e6c6739b1f6d3fa0d07d0e895e480f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beb2866ad646a4731fb9007aee1a16dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c34b32b1ea66605279a2ac0b74514239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b5805ec7af0d88bfb6e9accc93507e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c34b32b1ea66605279a2ac0b74514239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b5805ec7af0d88bfb6e9accc93507e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c34b32b1ea66605279a2ac0b74514239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b5805ec7af0d88bfb6e9accc93507e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c34b32b1ea66605279a2ac0b74514239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b5805ec7af0d88bfb6e9accc93507e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c34b32b1ea66605279a2ac0b74514239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b5805ec7af0d88bfb6e9accc93507e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c34b32b1ea66605279a2ac0b74514239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b5805ec7af0d88bfb6e9accc93507e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ac5b60e32655f106e5cb57a1ce901423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c6c010d8fe72871bd214119eeaf22c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_832094585afa456c071fc94e07acf7f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 288, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e3ae15e3947441f0b2eccb4335fcefde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_832094585afa456c071fc94e07acf7f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4a5f445529bbb09c09983c1795e02ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6060a67461a022847a401eb2a69a853
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4a5f445529bbb09c09983c1795e02ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6060a67461a022847a401eb2a69a853
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4a5f445529bbb09c09983c1795e02ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6060a67461a022847a401eb2a69a853
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4a5f445529bbb09c09983c1795e02ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6060a67461a022847a401eb2a69a853
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4a5f445529bbb09c09983c1795e02ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6060a67461a022847a401eb2a69a853
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4a5f445529bbb09c09983c1795e02ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6060a67461a022847a401eb2a69a853
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c48945a461a948ca7b932b45da8ac59d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db9bd9300274c0e9cd7ed6222b6dde73
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
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


class TestPrimitiveOp_619109e79a4fa4d8925f569959669b82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58898b83d91a13f5b58fc4b0af582f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_619109e79a4fa4d8925f569959669b82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58898b83d91a13f5b58fc4b0af582f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_619109e79a4fa4d8925f569959669b82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58898b83d91a13f5b58fc4b0af582f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_619109e79a4fa4d8925f569959669b82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58898b83d91a13f5b58fc4b0af582f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_619109e79a4fa4d8925f569959669b82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58898b83d91a13f5b58fc4b0af582f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_619109e79a4fa4d8925f569959669b82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58898b83d91a13f5b58fc4b0af582f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1646d7e6b4bade65a8319cfe9b6cdc85(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 40, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_14ea55973462bd871fe9e3f3fea2e28d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1646d7e6b4bade65a8319cfe9b6cdc85
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9519ce64a4c630a9a2dac537c370bbba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9995740a54c377c85f8688c7f37261f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1fa9dff567c1688bac5e9fdd77fdc561(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28e2137aee26b5ebeb884e5eee400c3c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.2846282720565796]], [[0.13444772362709045]], [[-0.7853584289550781]], [[-0.036956459283828735]], [[0.7129179835319519]], [[1.0102742910385132]], [[-0.04063612222671509]], [[0.28942447900772095]], [[-0.4624338150024414]], [[-0.09594550728797913]], [[2.294184684753418]], [[-0.39676931500434875]], [[-0.33506232500076294]], [[0.6328108906745911]], [[-0.1834728717803955]], [[-1.2857528924942017]], [[0.7599840760231018]], [[-1.507845163345337]], [[-0.5636314153671265]], [[0.7218278646469116]], [[1.0863370895385742]], [[0.8413982391357422]], [[-0.18167906999588013]], [[0.7051531076431274]], [[0.07017123699188232]], [[-1.510962724685669]], [[-1.3861486911773682]], [[-0.5816459059715271]], [[-0.005456916987895966]], [[-0.13362717628479004]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_5b5114be98fcf2b0b8296396d8151b3d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 22, 33], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee42a00d1f1e2933a69a4d8eaab2b1f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b5114be98fcf2b0b8296396d8151b3d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_8287e23e4a4c77121e2a4cb2ca1eabe9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 96, 109, 109], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_002acfa930bb9594c821d91b91abffd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8287e23e4a4c77121e2a4cb2ca1eabe9
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 109, 109], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_5bd151cfae2833db46bf7c87183eaef7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 16, 54, 54], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_195aefa3729dd4aeb8b28081060433a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bd151cfae2833db46bf7c87183eaef7
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_2fa709534109665523c5b5c2d27e04c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 64, 54, 54], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a3071d14b1d5ed956ea243a0c813d16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fa709534109665523c5b5c2d27e04c6
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7a3071d14b1d5ed956ea243a0c813d16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fa709534109665523c5b5c2d27e04c6
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_195aefa3729dd4aeb8b28081060433a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bd151cfae2833db46bf7c87183eaef7
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7a3071d14b1d5ed956ea243a0c813d16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fa709534109665523c5b5c2d27e04c6
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7a3071d14b1d5ed956ea243a0c813d16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fa709534109665523c5b5c2d27e04c6
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_36652724dcc63a729ff5bb720825f10e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 32, 54, 54], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d57bf9bff0a960bf03cf15fdf2f6b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36652724dcc63a729ff5bb720825f10e
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_389c420a3ef560d64a58be338e848a10(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 128, 54, 54], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1fe085cd068e03bf56d7a7d70a765c5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_389c420a3ef560d64a58be338e848a10
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1fe085cd068e03bf56d7a7d70a765c5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_389c420a3ef560d64a58be338e848a10
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_80ad7fc805f3f895cc88b5bb2fcea34c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 32, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a4d356ef9acebdd4f03fb26c80362037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ad7fc805f3f895cc88b5bb2fcea34c
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_a8188434df453797179524710fd4cb05(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 128, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9fd20549a8e94d22addfee4ad4506b9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8188434df453797179524710fd4cb05
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9fd20549a8e94d22addfee4ad4506b9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8188434df453797179524710fd4cb05
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_00e47cfab2636be5d98bf837f0581e07(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 48, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b830c670d9361c566abe6e0bc93a1c30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00e47cfab2636be5d98bf837f0581e07
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_2ed36699ffe332ed4debe73c94b59611(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 192, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2fe7ef931a45e086f0fbd15bb6755baa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed36699ffe332ed4debe73c94b59611
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2fe7ef931a45e086f0fbd15bb6755baa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed36699ffe332ed4debe73c94b59611
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b830c670d9361c566abe6e0bc93a1c30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00e47cfab2636be5d98bf837f0581e07
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2fe7ef931a45e086f0fbd15bb6755baa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed36699ffe332ed4debe73c94b59611
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2fe7ef931a45e086f0fbd15bb6755baa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed36699ffe332ed4debe73c94b59611
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_81b8d7025e78af386908981667cc86d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 64, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fecdcc54d15f4f6c7b91e8261feb3ae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81b8d7025e78af386908981667cc86d9
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_95a497b085630d5d7ac66ad6b34b10fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 256, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c6a0ad1da8c2d56d692b218ed439c445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95a497b085630d5d7ac66ad6b34b10fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c6a0ad1da8c2d56d692b218ed439c445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95a497b085630d5d7ac66ad6b34b10fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_3d12d01144a49a4737d74441cc9c55d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 64, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa4b8ce8e9761520e4e80b7c672418c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d12d01144a49a4737d74441cc9c55d8
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_4612f453f1b860e9d3f6ac54cbfc649a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 256, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ce0f0a87bf0618bf7f21caeba2c80f25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4612f453f1b860e9d3f6ac54cbfc649a
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ce0f0a87bf0618bf7f21caeba2c80f25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4612f453f1b860e9d3f6ac54cbfc649a
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_eeabc905bbf7761a155d2d0387dcf6fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1000, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8b5320885096263d07c6f0db5bed41d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeabc905bbf7761a155d2d0387dcf6fb
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_4a5ae7235f32662ddca4a7b52acab216(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 56, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bfd4909dcaa6e9c52457e93f7fbc18c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ae7235f32662ddca4a7b52acab216
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
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


class TestPrimitiveOp_b1f4d0e4bd69dc9f6325b9da47685660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e46c4d974bba9fa83ba73424a9c938bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1f4d0e4bd69dc9f6325b9da47685660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e46c4d974bba9fa83ba73424a9c938bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1f4d0e4bd69dc9f6325b9da47685660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e46c4d974bba9fa83ba73424a9c938bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1f4d0e4bd69dc9f6325b9da47685660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e46c4d974bba9fa83ba73424a9c938bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1f4d0e4bd69dc9f6325b9da47685660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e46c4d974bba9fa83ba73424a9c938bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1f4d0e4bd69dc9f6325b9da47685660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e46c4d974bba9fa83ba73424a9c938bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0775cd03df39a90fc99da78c4446a890(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_75f8de23d3f623fa1c3a6273ae4c3487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0775cd03df39a90fc99da78c4446a890
    def get_inputs(self):
        return [
            paddle.uniform([145, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_8e6411976882ad8aca25fe147ce2872f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[145, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d9ed478842302a62eb6e5801af416f9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e6411976882ad8aca25fe147ce2872f
    def get_inputs(self):
        return [
            paddle.uniform([145, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_95c01308904c40adca4c39f8e41f913e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_637f37fde95c5cabe8eb679aee3f3fd5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.1634543389081955]], [[0.7411292195320129]], [[-0.21232646703720093]], [[-1.036484718322754]], [[0.09134286642074585]], [[0.03124326467514038]], [[-1.055812120437622]], [[-0.5866613388061523]], [[0.8437510132789612]], [[0.17472049593925476]], [[-0.3439794182777405]], [[0.5364038944244385]], [[0.11725880205631256]], [[-0.878522515296936]], [[0.11486777663230896]], [[-0.2581847310066223]], [[0.5376848578453064]], [[0.9653182029724121]], [[-0.045372575521469116]], [[1.272841453552246]], [[0.9311769604682922]], [[0.4470592737197876]], [[-1.529889464378357]], [[-0.38696855306625366]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f4e145919a611c64a0975699bc7e4db6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28e2137aee26b5ebeb884e5eee400c3c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.04963985085487366]], [[-0.36315953731536865]], [[-1.3849685192108154]], [[-1.6081104278564453]], [[-0.7306016683578491]], [[0.3566800653934479]], [[-0.46621954441070557]], [[-0.9301321506500244]], [[0.11744210124015808]], [[-0.9655279517173767]], [[1.0388224124908447]], [[0.8177359700202942]], [[-0.46503010392189026]], [[-0.26188021898269653]], [[0.4530542492866516]], [[0.15016496181488037]], [[-0.5123621225357056]], [[1.5214747190475464]], [[-0.653536319732666]], [[-0.3717188239097595]], [[-0.1044144332408905]], [[0.5916125178337097]], [[0.051314398646354675]], [[-1.7697014808654785]], [[-1.2300527095794678]], [[0.30481648445129395]], [[1.6177029609680176]], [[-0.6559355854988098]], [[-1.022160291671753]], [[-0.42513251304626465]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e3ae15e3947441f0b2eccb4335fcefde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_832094585afa456c071fc94e07acf7f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c7a2a3b198e1de8e9e734caf491bc5d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ca72b5c3f73e21641e1f66a07f4e6fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fd4d2724f21366174dcc0d4f659c95cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1a888e2572e2939c956dd8a8ab8d6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1e73e0b4e3ba0de55de2c1e175d960d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_236421ce78df60215bd2a7c878c8d700
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_16a441208348730a5c7fe93d9bf10fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c9986f333ff64c7e35a9670e33c0a58
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_0a83ef083872fc8bcae2b9aeb70b7ddd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af83c9e4047e9bb4e9c450f91b80187d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a83ef083872fc8bcae2b9aeb70b7ddd
    def get_inputs(self):
        return [
            paddle.uniform([10, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_c0e3e1a35ed0e80e53314dc19a8dee9d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d8e60e6e16c10c44a4e35d32d8deacb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0e3e1a35ed0e80e53314dc19a8dee9d
    def get_inputs(self):
        return [
            paddle.uniform([10, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_16a441208348730a5c7fe93d9bf10fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c9986f333ff64c7e35a9670e33c0a58
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e3d79641638b3016d46a72068daab85b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b49aa34510522763d72a6c0031adcc7
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2efd6c58da2618552851b9580eb269ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccaad1c8b27d535fa8f01d6d25382881
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_16a441208348730a5c7fe93d9bf10fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c9986f333ff64c7e35a9670e33c0a58
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_288ccca0da1e352292f75ed334ba7c07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97ad36a27c2e642436ff8e15a33cd1a8
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.7100671529769897]], [[0.15242338180541992]], [[0.7567925453186035]], [[0.7487736940383911]], [[-0.6334264278411865]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7d84a8b45d902ba5fdcec62f9677a86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_411bb89e19b1a9d716ca821e15c913b8
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.7679162621498108]], [[1.4272069931030273]], [[1.1337499618530273]], [[1.430799961090088]], [[-1.1786308288574219]], [[-0.2300892472267151]], [[-0.11848604679107666]], [[-0.6000401377677917]], [[0.3070733845233917]], [[-1.3044531345367432]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_05addcfb8864d311ee5cbdec1fa46b5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ad42ceb01a9681f80e18af1bf83da66
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.03599497675895691]], [[-0.7396931648254395]], [[0.6570923328399658]], [[-0.8015350699424744]], [[1.8955659866333008]], [[0.028047800064086914]], [[-0.811128556728363]], [[1.833953619003296]], [[2.047973155975342]], [[1.3527365922927856]], [[-0.18485215306282043]], [[-1.9614239931106567]], [[-1.6231920719146729]], [[-1.5151324272155762]], [[-0.7188615202903748]], [[0.578149676322937]], [[1.1600956916809082]], [[-0.45801645517349243]], [[-1.3709503412246704]], [[-1.463099479675293]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0d8bad7df106ca61acc322e3618aaace(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc20a094848a67e308d963ec5296b27d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_f3e637d68ec4a14259710b809a192211(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 24, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f09ad3770e0b3dcc6c9f61b3ddd79e3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3e637d68ec4a14259710b809a192211
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_d4a51bac2ea0ef8f525980cf978bab2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 96, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5cf7495bf1b6bc734309ed1831b1d0c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4a51bac2ea0ef8f525980cf978bab2c
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f09ad3770e0b3dcc6c9f61b3ddd79e3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3e637d68ec4a14259710b809a192211
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5cf7495bf1b6bc734309ed1831b1d0c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4a51bac2ea0ef8f525980cf978bab2c
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f09ad3770e0b3dcc6c9f61b3ddd79e3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3e637d68ec4a14259710b809a192211
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5cf7495bf1b6bc734309ed1831b1d0c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4a51bac2ea0ef8f525980cf978bab2c
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f09ad3770e0b3dcc6c9f61b3ddd79e3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3e637d68ec4a14259710b809a192211
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5cf7495bf1b6bc734309ed1831b1d0c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4a51bac2ea0ef8f525980cf978bab2c
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_c2451ebaec26fb8eed39f0e1076c74aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 6, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_44216e77049e6818b6011bb7c512c1e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2451ebaec26fb8eed39f0e1076c74aa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.5621795654296875]], [[12.922218322753906]], [[-24.337650299072266]], [[-6.47609281539917]], [[13.357481002807617]], [[34.35028839111328]]], [[[1.825583577156067]], [[-12.206561088562012]], [[-1.2362172603607178]], [[34.30895233154297]], [[-18.37959861755371]], [[-19.932361602783203]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f09ad3770e0b3dcc6c9f61b3ddd79e3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3e637d68ec4a14259710b809a192211
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_17427b1064adbf94e7df8a015fbdddea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2451ebaec26fb8eed39f0e1076c74aa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[28.668880462646484]], [[-3.6743435859680176]], [[-16.643526077270508]], [[-16.576202392578125]], [[-34.483150482177734]], [[22.416324615478516]]], [[[12.590935707092285]], [[11.860464096069336]], [[32.43117904663086]], [[-8.74240779876709]], [[-28.7945499420166]], [[8.308571815490723]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f09ad3770e0b3dcc6c9f61b3ddd79e3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3e637d68ec4a14259710b809a192211
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d843e69bd2c1849ffaece61a5c49e41a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2451ebaec26fb8eed39f0e1076c74aa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[19.462419509887695]], [[38.729339599609375]], [[-10.175227165222168]], [[-29.6886043548584]], [[-8.977588653564453]], [[-32.49509811401367]]], [[[-3.1796822547912598]], [[7.059480667114258]], [[-37.31666564941406]], [[16.53364372253418]], [[-29.782033920288086]], [[0.7585366368293762]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f09ad3770e0b3dcc6c9f61b3ddd79e3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3e637d68ec4a14259710b809a192211
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f36dfe546dc41b3cee773b54eb195b0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2451ebaec26fb8eed39f0e1076c74aa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[23.48723030090332]], [[-33.11387634277344]], [[-3.0140419006347656]], [[-12.411575317382812]], [[23.337543487548828]], [[-17.387676239013672]]], [[[8.882867813110352]], [[-11.679582595825195]], [[-4.12380313873291]], [[13.209617614746094]], [[11.855643272399902]], [[10.772425651550293]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f09ad3770e0b3dcc6c9f61b3ddd79e3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3e637d68ec4a14259710b809a192211
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e3ae15e3947441f0b2eccb4335fcefde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_832094585afa456c071fc94e07acf7f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37fca934ffb81e15faed219aebc0f8ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d991bfc144a40c4dc77665a8845b8737
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d4c4b5cebb472b6e6c74fbdc247d046(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb66f66d335d8131349f351aea004f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fd4d2724f21366174dcc0d4f659c95cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1a888e2572e2939c956dd8a8ab8d6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1e73e0b4e3ba0de55de2c1e175d960d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_236421ce78df60215bd2a7c878c8d700
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2efd6c58da2618552851b9580eb269ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccaad1c8b27d535fa8f01d6d25382881
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c48945a461a948ca7b932b45da8ac59d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db9bd9300274c0e9cd7ed6222b6dde73
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9519ce64a4c630a9a2dac537c370bbba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9995740a54c377c85f8688c7f37261f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c7a2a3b198e1de8e9e734caf491bc5d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ca72b5c3f73e21641e1f66a07f4e6fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2efd6c58da2618552851b9580eb269ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccaad1c8b27d535fa8f01d6d25382881
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
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


class TestPrimitiveOp_dc34ec2d9548aec6f1b0e56225342282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_483c8d6e38258822c3c09154d493ccd3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dc34ec2d9548aec6f1b0e56225342282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_483c8d6e38258822c3c09154d493ccd3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dc34ec2d9548aec6f1b0e56225342282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_483c8d6e38258822c3c09154d493ccd3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dc34ec2d9548aec6f1b0e56225342282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_483c8d6e38258822c3c09154d493ccd3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dc34ec2d9548aec6f1b0e56225342282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_483c8d6e38258822c3c09154d493ccd3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dc34ec2d9548aec6f1b0e56225342282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_483c8d6e38258822c3c09154d493ccd3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_d9bb51f74d63ca8355e93d40c185595d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31af42573a883d58f3432c9419cf8682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d9bb51f74d63ca8355e93d40c185595d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31af42573a883d58f3432c9419cf8682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d9bb51f74d63ca8355e93d40c185595d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31af42573a883d58f3432c9419cf8682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d9bb51f74d63ca8355e93d40c185595d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31af42573a883d58f3432c9419cf8682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d9bb51f74d63ca8355e93d40c185595d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31af42573a883d58f3432c9419cf8682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d9bb51f74d63ca8355e93d40c185595d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31af42573a883d58f3432c9419cf8682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7944fc46a365e950267c4ace673b2d80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4188ede58228c460b705589b1a6124f0
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.5106748342514038]], [[-0.08315271139144897]], [[0.6998074054718018]], [[-0.3930266499519348]], [[0.3330589830875397]], [[-0.19780616462230682]], [[-0.3086894452571869]], [[-1.2034721374511719]], [[-0.04559686779975891]], [[-0.2661136984825134]], [[0.3996647000312805]], [[-0.004144340753555298]], [[0.5292736291885376]], [[-0.5092132091522217]], [[-0.9191134572029114]], [[-0.08649718761444092]], [[1.677173137664795]], [[-0.40747299790382385]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac5b60e32655f106e5cb57a1ce901423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c6c010d8fe72871bd214119eeaf22c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7b3c87b322c24101810f39d870848d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0ffdd5bf7236aed70be849b15ec5d41
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1c8fe4b5b59b3da7308318058667baf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41445b01980185f9e9f7e623d502fb75
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
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


class TestPrimitiveOp_77f1220cdacdfacb10b47b78750e7c59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bda4b337647876c6122658aac83ee09d
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.4661974608898163]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.23364883661270142]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_1c02dac282dbe6ea0a91801a586b5e12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bda4b337647876c6122658aac83ee09d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10313171148300171]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.22258034348487854]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_4bfb732d7d3dd4717c5c50fe40c5f739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bda4b337647876c6122658aac83ee09d
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.7297652959823608]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_248ef741916aa54db31c0222002b7b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bda4b337647876c6122658aac83ee09d
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.23912909626960754]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_4d79c61dadbe1d60383cafa3015c3d6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bda4b337647876c6122658aac83ee09d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.009030580520629883]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.49611642956733704]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_b8826eb4eea9fb62b5f1879f58283d97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bda4b337647876c6122658aac83ee09d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17337018251419067]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.13599738478660583]], dtype='float32').reshape([1, 1]),
        ]


class PrimitiveOp_280d782e3763959856e2f6b2e1d75b9e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_68ee931edc98016466bca6f14d40c3d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_280d782e3763959856e2f6b2e1d75b9e
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7b3c87b322c24101810f39d870848d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0ffdd5bf7236aed70be849b15ec5d41
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1c8fe4b5b59b3da7308318058667baf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41445b01980185f9e9f7e623d502fb75
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a1604ae7c75c4f2894028b5b55b817ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28e2137aee26b5ebeb884e5eee400c3c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-1.899590253829956]], [[0.9426041841506958]], [[-0.7342880964279175]], [[-0.13855992257595062]], [[-0.4397205710411072]], [[-0.011864662170410156]], [[0.22731083631515503]], [[-0.038987383246421814]], [[-0.32013624906539917]], [[0.4540535509586334]], [[-0.24005156755447388]], [[0.026912927627563477]], [[-0.4953871965408325]], [[-0.38113558292388916]], [[-0.0673292800784111]], [[-1.3439576625823975]], [[-2.371751070022583]], [[0.3707817792892456]], [[0.9890931844711304]], [[0.07847198843955994]], [[0.3483148217201233]], [[-1.5437400341033936]], [[-0.7459312677383423]], [[1.460159182548523]], [[0.8118617534637451]], [[-0.8307193517684937]], [[1.343743085861206]], [[0.32366853952407837]], [[0.8362864255905151]], [[-0.27731043100357056]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d9bb51f74d63ca8355e93d40c185595d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31af42573a883d58f3432c9419cf8682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d9bb51f74d63ca8355e93d40c185595d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31af42573a883d58f3432c9419cf8682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d9bb51f74d63ca8355e93d40c185595d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31af42573a883d58f3432c9419cf8682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d9bb51f74d63ca8355e93d40c185595d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31af42573a883d58f3432c9419cf8682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d9bb51f74d63ca8355e93d40c185595d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31af42573a883d58f3432c9419cf8682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d9bb51f74d63ca8355e93d40c185595d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31af42573a883d58f3432c9419cf8682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c48945a461a948ca7b932b45da8ac59d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db9bd9300274c0e9cd7ed6222b6dde73
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_1cc08b4edac156406f3c290cd484add4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e0607eac78e4d2940b2668ad48b3fea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1cc08b4edac156406f3c290cd484add4
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8e0607eac78e4d2940b2668ad48b3fea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1cc08b4edac156406f3c290cd484add4
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d17cdfddf793e6d9c3bede675b97d546(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f60b8e268f203712cdfd228e8f3ad2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d17cdfddf793e6d9c3bede675b97d546(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f60b8e268f203712cdfd228e8f3ad2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d17cdfddf793e6d9c3bede675b97d546(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f60b8e268f203712cdfd228e8f3ad2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d17cdfddf793e6d9c3bede675b97d546(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f60b8e268f203712cdfd228e8f3ad2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d17cdfddf793e6d9c3bede675b97d546(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f60b8e268f203712cdfd228e8f3ad2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d17cdfddf793e6d9c3bede675b97d546(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f60b8e268f203712cdfd228e8f3ad2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e3ae15e3947441f0b2eccb4335fcefde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_832094585afa456c071fc94e07acf7f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_8ea67e7ff5fa37ef3d36336dddc62f60(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 11, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f738fe60847983a0e7ca9775cd6a6b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ea67e7ff5fa37ef3d36336dddc62f60
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fd4d2724f21366174dcc0d4f659c95cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1a888e2572e2939c956dd8a8ab8d6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2efd6c58da2618552851b9580eb269ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccaad1c8b27d535fa8f01d6d25382881
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5e4164e3d60b6586339e17ebf8018de2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0484c9d013b12a30430fcd0c3e022673
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_f8801709d8c241df87ed3126f4531979(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 28, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df304a91b6f949ab08e72353998f74a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8801709d8c241df87ed3126f4531979
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7af2ae20c5acafe2483461f941b66fc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28e2137aee26b5ebeb884e5eee400c3c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.6833476424217224]], [[-1.1979091167449951]], [[-1.3699426651000977]], [[-1.4419666528701782]], [[0.5541517734527588]], [[-1.0024560689926147]], [[-1.3260233402252197]], [[1.2173478603363037]], [[-0.473552942276001]], [[0.2573644518852234]], [[-0.2996983528137207]], [[0.026211872696876526]], [[0.05123484134674072]], [[-1.4185924530029297]], [[0.8587384819984436]], [[-0.35836660861968994]], [[0.3155171573162079]], [[-0.2104724645614624]], [[0.6400618553161621]], [[-1.2706689834594727]], [[1.6130930185317993]], [[-0.7225378751754761]], [[-0.08711212873458862]], [[-0.20906329154968262]], [[0.5319129228591919]], [[1.015777587890625]], [[-1.3753440380096436]], [[0.1746717244386673]], [[0.6958909630775452]], [[1.7452642917633057]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_10f8e5f4893ade488356f91a8b3846e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c31399550ef202d07270f8162e49c66
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10f8e5f4893ade488356f91a8b3846e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c31399550ef202d07270f8162e49c66
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10f8e5f4893ade488356f91a8b3846e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c31399550ef202d07270f8162e49c66
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10f8e5f4893ade488356f91a8b3846e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c31399550ef202d07270f8162e49c66
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10f8e5f4893ade488356f91a8b3846e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c31399550ef202d07270f8162e49c66
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10f8e5f4893ade488356f91a8b3846e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c31399550ef202d07270f8162e49c66
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_17355a61c12305c604297e08b7d9b5f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_17355a61c12305c604297e08b7d9b5f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_187518afb2e7732389c014d8a702fa4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_187518afb2e7732389c014d8a702fa4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_17355a61c12305c604297e08b7d9b5f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_17355a61c12305c604297e08b7d9b5f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8e502848818efa3ab4c64982c03bb783(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 11109, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9573da0022e01a76f5a7cb0bed348473(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e502848818efa3ab4c64982c03bb783
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 11109, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_864e3ab6b4ac8699da611953dd512202(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4188ede58228c460b705589b1a6124f0
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.2836771011352539]], [[-0.23993463814258575]], [[-0.23711249232292175]], [[0.13185271620750427]], [[0.7390395998954773]], [[-0.8038957118988037]], [[-1.1033554077148438]], [[-0.7345924377441406]], [[-0.9306720495223999]], [[0.06498736143112183]], [[-0.3362985849380493]], [[-0.42814794182777405]], [[-0.1628945916891098]], [[-0.9774215817451477]], [[0.03604963421821594]], [[1.0091943740844727]], [[-0.23625588417053223]], [[1.280714511871338]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac5b60e32655f106e5cb57a1ce901423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c6c010d8fe72871bd214119eeaf22c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_16a441208348730a5c7fe93d9bf10fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c9986f333ff64c7e35a9670e33c0a58
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1ac5217bef83b547fe873b9b1a5f4400(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76211934d2a84fd2ee5cf8ea4fc0ccfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1ac5217bef83b547fe873b9b1a5f4400(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76211934d2a84fd2ee5cf8ea4fc0ccfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1ac5217bef83b547fe873b9b1a5f4400(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76211934d2a84fd2ee5cf8ea4fc0ccfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1ac5217bef83b547fe873b9b1a5f4400(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76211934d2a84fd2ee5cf8ea4fc0ccfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1ac5217bef83b547fe873b9b1a5f4400(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76211934d2a84fd2ee5cf8ea4fc0ccfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1ac5217bef83b547fe873b9b1a5f4400(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76211934d2a84fd2ee5cf8ea4fc0ccfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0e07c6f0e4ed41d7b80586679bf092d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 5, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_62fe15b9c5b45431b3a1b6c2303ea1a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e07c6f0e4ed41d7b80586679bf092d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_16a441208348730a5c7fe93d9bf10fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c9986f333ff64c7e35a9670e33c0a58
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1e73e0b4e3ba0de55de2c1e175d960d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_236421ce78df60215bd2a7c878c8d700
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_dc34ec2d9548aec6f1b0e56225342282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_483c8d6e38258822c3c09154d493ccd3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dc34ec2d9548aec6f1b0e56225342282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_483c8d6e38258822c3c09154d493ccd3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dc34ec2d9548aec6f1b0e56225342282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_483c8d6e38258822c3c09154d493ccd3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dc34ec2d9548aec6f1b0e56225342282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_483c8d6e38258822c3c09154d493ccd3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dc34ec2d9548aec6f1b0e56225342282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_483c8d6e38258822c3c09154d493ccd3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dc34ec2d9548aec6f1b0e56225342282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_483c8d6e38258822c3c09154d493ccd3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3271cbd7ddc973591effbb5d028c49e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3271cbd7ddc973591effbb5d028c49e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_74f3134b9d32ca9787e6a6a07406bcf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_74f3134b9d32ca9787e6a6a07406bcf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3271cbd7ddc973591effbb5d028c49e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3271cbd7ddc973591effbb5d028c49e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a78664321cae7e962efe7dc3a55e89f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2100, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d34e31a33ad4bd6d0b5b11050dc196e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a78664321cae7e962efe7dc3a55e89f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2100, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b2174f9b8cec5e60b28d4a1bad6cdf22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 36, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_46bb2565cb498d4814daded00b561779(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2174f9b8cec5e60b28d4a1bad6cdf22
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7b18e13f08ec7601bd78d0fb61bbfbad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7af53569cf6eb3e207d2d6b73a2fbce9
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c380db5209030645617934cdbb8f98c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc1ee3c988ffeebceb71659ab5d57bab
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.7226943969726562]], [[-0.309501051902771]], [[0.39059337973594666]], [[0.14509093761444092]], [[0.432636022567749]], [[1.1142088174819946]], [[-0.4448205828666687]], [[-0.5387935638427734]], [[1.390832781791687]], [[-0.6262253522872925]], [[0.4991160035133362]]]], dtype='float32').reshape([1, 11, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b6657cf91c94b461da2686f9bb389f2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_622954dcb771c1e5a4bf44699528e776
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7b3c87b322c24101810f39d870848d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0ffdd5bf7236aed70be849b15ec5d41
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1c8fe4b5b59b3da7308318058667baf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41445b01980185f9e9f7e623d502fb75
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d3e136d01894611d3076edbbe6eff4c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d3e136d01894611d3076edbbe6eff4c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cb1b850eb0e779044e9eaf95bb66cc78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cb1b850eb0e779044e9eaf95bb66cc78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d3e136d01894611d3076edbbe6eff4c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d3e136d01894611d3076edbbe6eff4c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b172a8f3c86e3e3e2baffe7f7b915644(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d132eacb1e126448f2ecf9489c59ec1e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_16a441208348730a5c7fe93d9bf10fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c9986f333ff64c7e35a9670e33c0a58
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_089b3d14e8fcca40c999d21f22a6a4ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_36b27d5aeed32949a06cbe90f2292d23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_089b3d14e8fcca40c999d21f22a6a4ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0d8bad7df106ca61acc322e3618aaace(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc20a094848a67e308d963ec5296b27d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3f7d1b06be540b4a918d2acf245ad3a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cf47f1fb1d3a0c430b428dfa0bdca51
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2efd6c58da2618552851b9580eb269ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccaad1c8b27d535fa8f01d6d25382881
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_beb1da3291be52cff8b38e91f6f0ff9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aaf2f758df5adbe4e39f7a29b963dcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_beb1da3291be52cff8b38e91f6f0ff9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aaf2f758df5adbe4e39f7a29b963dcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_beb1da3291be52cff8b38e91f6f0ff9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aaf2f758df5adbe4e39f7a29b963dcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_beb1da3291be52cff8b38e91f6f0ff9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aaf2f758df5adbe4e39f7a29b963dcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_beb1da3291be52cff8b38e91f6f0ff9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aaf2f758df5adbe4e39f7a29b963dcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_beb1da3291be52cff8b38e91f6f0ff9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aaf2f758df5adbe4e39f7a29b963dcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_04f07737e2cb8d3ec08fd00813ab493d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 112, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f554b8281c85dc9bc63c96e2e35c559(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04f07737e2cb8d3ec08fd00813ab493d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_70cb339b3f7a24cedc26cc1de6728434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aaf550957e26b3fa1ff256401484b75d
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8f70172e1c6e2848fc88765e9f25221f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac1fc699d1242d3cc0e20637a40547a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_44dcb5bb08e100777178bc9cfeee7d2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28e2137aee26b5ebeb884e5eee400c3c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.9045238494873047]], [[0.8758697509765625]], [[-0.1450267732143402]], [[0.6795176267623901]], [[0.6280986070632935]], [[-0.1595645546913147]], [[-0.6543539762496948]], [[0.1574198603630066]], [[-0.22853633761405945]], [[1.6612236499786377]], [[0.5316828489303589]], [[0.10284401476383209]], [[-0.308243989944458]], [[-0.16900894045829773]], [[0.17386935651302338]], [[-0.2600306272506714]], [[-0.27606552839279175]], [[0.5136259198188782]], [[0.35588428378105164]], [[-1.6710894107818604]], [[1.3968391418457031]], [[-0.12520447373390198]], [[1.6691827774047852]], [[0.6804451942443848]], [[0.808747410774231]], [[0.7789115905761719]], [[1.3305268287658691]], [[-1.511985182762146]], [[1.1124943494796753]], [[0.266329824924469]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_63000b7df1d16840e48dcf697446473a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28e2137aee26b5ebeb884e5eee400c3c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.5169028043746948]], [[-0.5940489768981934]], [[0.330168217420578]], [[-0.9360513687133789]], [[0.5400761365890503]], [[0.08140048384666443]], [[-0.7100027799606323]], [[0.8852694034576416]], [[0.36968857049942017]], [[0.754610002040863]], [[-2.1558072566986084]], [[0.7669030427932739]], [[-0.29336869716644287]], [[-0.6976342797279358]], [[0.720740556716919]], [[0.128004252910614]], [[0.4152948558330536]], [[-0.9039360284805298]], [[0.5437854528427124]], [[-0.02358749508857727]], [[-0.005630612373352051]], [[-0.09195688366889954]], [[-0.07317730784416199]], [[-1.1363685131072998]], [[-1.2907341718673706]], [[-0.7487685680389404]], [[0.6259253025054932]], [[0.3917490243911743]], [[-0.7907663583755493]], [[0.17342838644981384]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4d4678a5f8c3c5700486fb42e49ef0ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19c4d48a4614377a160bef248e7b0e50
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0f86ac4a04aace26c9eedc8ad4db9e33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_361104abfdbefc900fe5bf4c5f7622a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e91ae64d5b069147970620069fdea92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc87839e84a9ac19e1f82b6ff576d99f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e91ae64d5b069147970620069fdea92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc87839e84a9ac19e1f82b6ff576d99f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e91ae64d5b069147970620069fdea92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc87839e84a9ac19e1f82b6ff576d99f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e91ae64d5b069147970620069fdea92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc87839e84a9ac19e1f82b6ff576d99f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e91ae64d5b069147970620069fdea92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc87839e84a9ac19e1f82b6ff576d99f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e91ae64d5b069147970620069fdea92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc87839e84a9ac19e1f82b6ff576d99f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3e0efb1d7fd1339f1c89f4fc3a5725d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 96, 109, 109], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e632cbfbfb390fe824f63bc867d08937(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0efb1d7fd1339f1c89f4fc3a5725d1
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 109, 109], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_2750e3b3c18ce46c3741785fd1e7f44d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 16, 54, 54], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c56f03571659e43a1efb5b5b9e26cf7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2750e3b3c18ce46c3741785fd1e7f44d
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_f52b727b2dfe0629db64325dc245b977(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 64, 54, 54], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9ccd14a7eaeeb052e89f78cef4af322f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52b727b2dfe0629db64325dc245b977
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9ccd14a7eaeeb052e89f78cef4af322f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52b727b2dfe0629db64325dc245b977
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c56f03571659e43a1efb5b5b9e26cf7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2750e3b3c18ce46c3741785fd1e7f44d
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9ccd14a7eaeeb052e89f78cef4af322f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52b727b2dfe0629db64325dc245b977
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9ccd14a7eaeeb052e89f78cef4af322f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52b727b2dfe0629db64325dc245b977
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_4ec67eaef549d9450254c8733ea690fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 32, 54, 54], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_32e64adaf30dfb23a5a51bd8be773803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ec67eaef549d9450254c8733ea690fd
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_0d0fc657731602f57470863ede461a19(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 128, 54, 54], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_06e64d917ecd40e19d1898abc0757c49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d0fc657731602f57470863ede461a19
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06e64d917ecd40e19d1898abc0757c49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d0fc657731602f57470863ede461a19
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_838d87c054c7753163d38e6c5129ea07(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 32, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f7435686e1e28ee18b57280c9daf2ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_838d87c054c7753163d38e6c5129ea07
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_4049f61c15e7181c1127f97c6b1c8156(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 128, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f0ad219ac530d800bc0eff003988cce5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4049f61c15e7181c1127f97c6b1c8156
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f0ad219ac530d800bc0eff003988cce5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4049f61c15e7181c1127f97c6b1c8156
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_21eeab2c11e99b599f53d3bab4493fc4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 48, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_33a57a4ffdad84577c2ee7a282e5c11b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21eeab2c11e99b599f53d3bab4493fc4
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_310499ae18a01fed1c793d2a6b687fa5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 192, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f5445b36146a9beeabd69577a57ee80e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_310499ae18a01fed1c793d2a6b687fa5
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f5445b36146a9beeabd69577a57ee80e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_310499ae18a01fed1c793d2a6b687fa5
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_33a57a4ffdad84577c2ee7a282e5c11b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21eeab2c11e99b599f53d3bab4493fc4
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f5445b36146a9beeabd69577a57ee80e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_310499ae18a01fed1c793d2a6b687fa5
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f5445b36146a9beeabd69577a57ee80e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_310499ae18a01fed1c793d2a6b687fa5
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_a87b3ec8eea95ed327971c8beeb55bda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 64, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ea0ed6af4613446123388e27b463d888(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a87b3ec8eea95ed327971c8beeb55bda
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_8b20bc6c29963b0e70a0428aa550ad6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 256, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e460eb18be5a8d79f4542ef91426c59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b20bc6c29963b0e70a0428aa550ad6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3e460eb18be5a8d79f4542ef91426c59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b20bc6c29963b0e70a0428aa550ad6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_ba2177b091843937400491dabb33c2cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 64, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b0f023e7f6858f7ab79638bea3619a49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba2177b091843937400491dabb33c2cc
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_f198698192fe98389db0c5596ebe5fe2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 256, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5dac37000def9e88cf35a1cf770c78a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f198698192fe98389db0c5596ebe5fe2
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5dac37000def9e88cf35a1cf770c78a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f198698192fe98389db0c5596ebe5fe2
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_3ec2823d9a55dee4250fe8207487173a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1000, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f9ba852057bd3587136d14cc879445d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ec2823d9a55dee4250fe8207487173a
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_6bcb2ae4c9d67edd7d6db4b5ac691282(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 39], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7586ad4f2f05a3ad3f3cddf492ba6305(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bcb2ae4c9d67edd7d6db4b5ac691282
    def get_inputs(self):
        return [
            paddle.uniform([1, 39], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_bd0a0579231c952a281b6ac9889b08bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 156], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f2d25e1e2c3d3fde241bc1b0031ae3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd0a0579231c952a281b6ac9889b08bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 156], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e99fad0a0918e14d16cae5affb4bdbba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ad42ceb01a9681f80e18af1bf83da66
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.27166926860809326]], [[-0.04478153586387634]], [[0.5807841420173645]], [[-0.11871391534805298]], [[-0.26164931058883667]], [[-0.26926252245903015]], [[-1.0515294075012207]], [[-1.3162676095962524]], [[0.4375103712081909]], [[0.19353048503398895]], [[0.4154050350189209]], [[-0.8079526424407959]], [[-0.19153326749801636]], [[-0.8560777306556702]], [[-0.8255999088287354]], [[0.24123263359069824]], [[0.2803671956062317]], [[-0.10375398397445679]], [[0.523106575012207]], [[0.02397763729095459]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_28e6c6739b1f6d3fa0d07d0e895e480f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beb2866ad646a4731fb9007aee1a16dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_e0c6b4f0da476914cfa8c080787973cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 96, 109, 109], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d50dc5fa8ca3697ba801837ece0330ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0c6b4f0da476914cfa8c080787973cf
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 109, 109], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_62ebe0150011a921a4d829b48489000f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 54, 54], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_847c58e698fbd3ab9cf496e78a82a3be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62ebe0150011a921a4d829b48489000f
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_773dd0e54ee27341f849f14d3f553392(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 64, 54, 54], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a0285a8ae4119692c80a31de31ac991c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_773dd0e54ee27341f849f14d3f553392
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a0285a8ae4119692c80a31de31ac991c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_773dd0e54ee27341f849f14d3f553392
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_847c58e698fbd3ab9cf496e78a82a3be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62ebe0150011a921a4d829b48489000f
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a0285a8ae4119692c80a31de31ac991c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_773dd0e54ee27341f849f14d3f553392
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a0285a8ae4119692c80a31de31ac991c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_773dd0e54ee27341f849f14d3f553392
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_d8eb014711aa8ee9968a935a7b0cd9be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 32, 54, 54], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_56106a662d675264d45f51a35c727367(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8eb014711aa8ee9968a935a7b0cd9be
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_b203312470bf096d9151f2e824042683(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 128, 54, 54], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5aa7be332416c06f1a2de47cc638f7c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b203312470bf096d9151f2e824042683
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5aa7be332416c06f1a2de47cc638f7c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b203312470bf096d9151f2e824042683
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_5b7a41d377856e2316ff50a7d1fb33f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 32, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_13644a9774fda22ec7be8bc38da825f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b7a41d377856e2316ff50a7d1fb33f6
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_a1bb04533e3a15f9dfd02d1a66897137(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 128, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3490e63ba24bad8622207740a294e3a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1bb04533e3a15f9dfd02d1a66897137
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3490e63ba24bad8622207740a294e3a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1bb04533e3a15f9dfd02d1a66897137
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_80c843ebd4048652b307b524f83fcc8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 48, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a6a6de59e79d560159c18dd5f918e890(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80c843ebd4048652b307b524f83fcc8d
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_1f5e49b8449e8f9719959d37291da61e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 192, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0eaeae393114553f218fbd6bf5f47366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f5e49b8449e8f9719959d37291da61e
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0eaeae393114553f218fbd6bf5f47366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f5e49b8449e8f9719959d37291da61e
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a6a6de59e79d560159c18dd5f918e890(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80c843ebd4048652b307b524f83fcc8d
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0eaeae393114553f218fbd6bf5f47366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f5e49b8449e8f9719959d37291da61e
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0eaeae393114553f218fbd6bf5f47366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f5e49b8449e8f9719959d37291da61e
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_2798df331746424ed59965fad8eaea42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 64, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_41a19bf90ca3513154a92f9c08c0535d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2798df331746424ed59965fad8eaea42
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_48c1296303595357dda269b5d39425d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 256, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_45185aa10960e84574354fe2bc005021(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48c1296303595357dda269b5d39425d6
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_45185aa10960e84574354fe2bc005021(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48c1296303595357dda269b5d39425d6
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_0f3acfac91628c5b339830814ed2f42a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 64, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_518b234e0a1a415517bc9f3fd36d0fdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f3acfac91628c5b339830814ed2f42a
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_17fe9c22c7e80c4b04ff525092998c9c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 256, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97ee694c66f6831fec943f79764018d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fe9c22c7e80c4b04ff525092998c9c
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_97ee694c66f6831fec943f79764018d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fe9c22c7e80c4b04ff525092998c9c
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_17667bed3f1a57628b9b58dc19152e5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1000, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07eba3d306145bec58a43dfcbc12a605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17667bed3f1a57628b9b58dc19152e5b
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5965acade62d94c251ff6f95471f6760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b72a2cd01483ae5035b0d59feeefb30b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.20314082503318787]], [[-0.12713158130645752]], [[0.3228570520877838]], [[0.9060093760490417]]]], dtype='float32').reshape([1, 4, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6203c5ede334397d326e6985bd2215c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e64e8082edc6c652cf6ce9093b30699
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.4468441307544708]], [[0.4903221130371094]], [[0.5349292755126953]], [[0.5423062443733215]], [[0.4022231101989746]], [[0.608171820640564]], [[0.5130035877227783]], [[0.4183555543422699]], [[0.5500997304916382]], [[0.43059056997299194]], [[0.5732475519180298]], [[0.43051764369010925]], [[0.5852348208427429]], [[0.37766554951667786]], [[0.4168500304222107]], [[0.5743144750595093]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_311ed8f3713fbc52e3f347a257c82389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_723f3308f2acc683d28af99dfab14672
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_9b965144fdb6428a939758f26b8266b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_826ff0fce8ffd4f998f89b2ddc057265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b965144fdb6428a939758f26b8266b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_511fac7318c04a6310921b8ea2a57193(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7295be19a5e047d75be3d0f66f96ca8d
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a502c3d4e4096931e916c9e99319c4a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12c5ad24f743cb81e8a0095a3dccae95
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c48945a461a948ca7b932b45da8ac59d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db9bd9300274c0e9cd7ed6222b6dde73
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9519ce64a4c630a9a2dac537c370bbba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9995740a54c377c85f8688c7f37261f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_b004d18ea12512dbfe82c78bb32a9b1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 12, 18], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8a3930127b88c5990d4c5185e57fadba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b004d18ea12512dbfe82c78bb32a9b1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9519ce64a4c630a9a2dac537c370bbba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9995740a54c377c85f8688c7f37261f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0153f0324fbe360bb910fdb118a7395e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08289d443e6f5b0e48d90155c48b479b
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_629724ccc231d72e797a61c9aeefb94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_467c4596305cfa8adfbad479de79a399
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_d2a9a3e2cd0e9a92b51cd29d85490dfa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8a38264ba6f2109ca1c021c93de2876e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2a9a3e2cd0e9a92b51cd29d85490dfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_28e6c6739b1f6d3fa0d07d0e895e480f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beb2866ad646a4731fb9007aee1a16dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9d5fd8fd3a119636747083813b00fddf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40aeb0fd2b2dd8fe956835b6bdd3594d
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_336ed71aeb8adbb19c6387773a5a09c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_637f37fde95c5cabe8eb679aee3f3fd5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.9140563011169434]], [[-10.813276290893555]], [[1.6344404220581055]], [[-7.3315863609313965]], [[-6.163364887237549]], [[-4.887502670288086]], [[2.055950880050659]], [[1.9200050830841064]], [[-5.387573719024658]], [[3.511977434158325]], [[8.426935195922852]], [[-8.396696090698242]], [[-2.7008159160614014]], [[2.2349138259887695]], [[0.33809706568717957]], [[5.134037017822266]], [[-4.046077728271484]], [[-1.9119130373001099]], [[7.593814849853516]], [[-7.686255931854248]], [[-0.12978854775428772]], [[-1.1978778839111328]], [[11.749096870422363]], [[-5.874687194824219]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ebd5511647ac551c53bc75e1f59a1670(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_637f37fde95c5cabe8eb679aee3f3fd5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.5660536289215088]], [[0.03172677755355835]], [[-0.1385921984910965]], [[4.274409294128418]], [[-0.647004246711731]], [[-1.7694905996322632]], [[1.1716480255126953]], [[-1.626310110092163]], [[0.5538054704666138]], [[-0.8722269535064697]], [[0.2983852028846741]], [[-0.8136069774627686]], [[-1.3460638523101807]], [[-2.126546621322632]], [[0.08267870545387268]], [[0.9491353034973145]], [[2.335836172103882]], [[0.3544660210609436]], [[-0.7586761713027954]], [[-1.025423288345337]], [[-1.0113656520843506]], [[0.6146766543388367]], [[0.19652509689331055]], [[1.0359610319137573]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_57ff9f6f7ec90265d26ed8fbe45b3368(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_637f37fde95c5cabe8eb679aee3f3fd5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.1219068765640259]], [[2.162184476852417]], [[-0.10391318798065186]], [[-0.11064350605010986]], [[0.9885352849960327]], [[0.39720088243484497]], [[1.4281201362609863]], [[-0.9665923118591309]], [[0.27802038192749023]], [[0.444979190826416]], [[-0.27016255259513855]], [[-0.8977764248847961]], [[0.22830790281295776]], [[0.4121161103248596]], [[0.4414874017238617]], [[-0.08370652794837952]], [[0.4102335572242737]], [[-0.28442609310150146]], [[-0.6733777523040771]], [[-0.012491390109062195]], [[-1.0813583135604858]], [[1.476287841796875]], [[-2.2605175971984863]], [[-0.9677222967147827]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c74fa5944d699f08e4f711909ef86597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_637f37fde95c5cabe8eb679aee3f3fd5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.5911619663238525]], [[-0.41236886382102966]], [[-1.8618955612182617]], [[0.8546162247657776]], [[-0.6592543721199036]], [[0.6390159130096436]], [[-0.6643391251564026]], [[0.9139282703399658]], [[0.47438400983810425]], [[-0.6970769166946411]], [[-0.5414532423019409]], [[-0.11311358213424683]], [[1.6374026536941528]], [[-1.009516716003418]], [[0.5147844552993774]], [[1.377902626991272]], [[0.557539701461792]], [[0.005938351154327393]], [[0.0029514431953430176]], [[0.11939981579780579]], [[0.3687751293182373]], [[0.34688952565193176]], [[-0.3067397475242615]], [[0.9982709884643555]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_01a51dacb32fd5e358bd00add9eacfae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af835a2f444c6ddee02661eb8678bf34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01a51dacb32fd5e358bd00add9eacfae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-20.078048706054688]], [[7.656277179718018]], [[-66.4404067993164]], [[18.713415145874023]], [[-34.27458190917969]], [[7.1543169021606445]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_40b155d7a6e89ac66c6204fc7888449d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_637f37fde95c5cabe8eb679aee3f3fd5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.42418748140335083]], [[1.0]], [[1.0]], [[-0.9511007070541382]], [[-1.362120509147644]], [[0.001982450485229492]], [[0.8842978477478027]], [[-1.0343555212020874]], [[-0.7261601686477661]], [[1.0]], [[-0.8052194118499756]], [[1.0]], [[0.06433439254760742]], [[0.7106450200080872]], [[1.0]], [[-1.0501649379730225]], [[0.41633346676826477]], [[0.9811534881591797]], [[-1.0150970220565796]], [[-0.3432319760322571]], [[1.0]], [[0.20667269825935364]], [[-0.6107468605041504]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_75a01a1dd6e66b386cbc80a673f22968(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01a51dacb32fd5e358bd00add9eacfae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[32.737998962402344]], [[-18.050966262817383]], [[-77.17453002929688]], [[28.64984893798828]], [[-50.996978759765625]], [[42.71753692626953]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_80baa52c9d2fc99a7d8da5b4a007488a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_637f37fde95c5cabe8eb679aee3f3fd5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-3.2338485717773438]], [[1.0]], [[-3.6019740104675293]], [[-3.1069061756134033]], [[-2.658082962036133]], [[0.8967139720916748]], [[0.9470024108886719]], [[1.0]], [[0.3660065829753876]], [[1.0]], [[0.6674299240112305]], [[1.0]], [[-0.102855384349823]], [[1.0]], [[1.0]], [[-5.380918025970459]], [[-5.348049640655518]], [[-0.25527942180633545]], [[1.0]], [[0.09769514203071594]], [[1.0]], [[-2.016575336456299]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_25dcc46c6aaaf0c722cfe741c597df67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01a51dacb32fd5e358bd00add9eacfae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[37.43060302734375]], [[-4.579713344573975]], [[35.8532600402832]], [[16.336278915405273]], [[3.615065097808838]], [[50.083160400390625]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fc9475351956109d7aa6d21569ff9e73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_637f37fde95c5cabe8eb679aee3f3fd5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.17788970470428467]], [[-1.8235797882080078]], [[-2.679997205734253]], [[1.0]], [[-0.47933429479599]], [[0.26931238174438477]], [[1.0]], [[-6.290741920471191]], [[1.0]], [[1.0]], [[-0.9013689756393433]], [[-0.503687858581543]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[0.7809153199195862]], [[-0.5418663024902344]], [[-5.004980087280273]], [[1.0]], [[-1.795461654663086]], [[0.313679039478302]], [[-0.9901344776153564]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fe3368f334c8556779571d4c01d66a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01a51dacb32fd5e358bd00add9eacfae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[19.893211364746094]], [[2.178400754928589]], [[-52.362159729003906]], [[-24.98861312866211]], [[29.63703727722168]], [[23.06300926208496]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f73833a07846b006d9553118879e41fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_637f37fde95c5cabe8eb679aee3f3fd5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.712058424949646]], [[-3.0649235248565674]], [[1.0]], [[-2.5518593788146973]], [[1.0]], [[1.0]], [[0.15969279408454895]], [[1.0]], [[-0.48222798109054565]], [[1.0]], [[1.0]], [[0.1638200581073761]], [[-4.006744861602783]], [[1.0]], [[-0.18115484714508057]], [[-0.6350057125091553]], [[-0.16288256645202637]], [[-1.4942678213119507]], [[1.0]], [[-1.3142420053482056]], [[-0.9569375514984131]], [[1.0]], [[1.0]], [[-0.5392221212387085]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_28e6c6739b1f6d3fa0d07d0e895e480f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beb2866ad646a4731fb9007aee1a16dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9d5fd8fd3a119636747083813b00fddf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40aeb0fd2b2dd8fe956835b6bdd3594d
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7b3c87b322c24101810f39d870848d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0ffdd5bf7236aed70be849b15ec5d41
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1c8fe4b5b59b3da7308318058667baf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41445b01980185f9e9f7e623d502fb75
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37fca934ffb81e15faed219aebc0f8ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d991bfc144a40c4dc77665a8845b8737
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d4c4b5cebb472b6e6c74fbdc247d046(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb66f66d335d8131349f351aea004f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c48945a461a948ca7b932b45da8ac59d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db9bd9300274c0e9cd7ed6222b6dde73
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_dad57e2d1da721bfcb04075e6b00860b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4188ede58228c460b705589b1a6124f0
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.013479523360729218]], [[0.5204800367355347]], [[0.14201512932777405]], [[0.09683763980865479]], [[0.13166996836662292]], [[0.3335575461387634]], [[-0.04575811326503754]], [[-0.5316265821456909]], [[-0.15804645419120789]], [[0.8252749443054199]], [[-1.1574952602386475]], [[-1.1142890453338623]], [[0.19097734987735748]], [[-0.7853107452392578]], [[0.4363425672054291]], [[0.6822408437728882]], [[0.6156995296478271]], [[0.4689646065235138]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac5b60e32655f106e5cb57a1ce901423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c6c010d8fe72871bd214119eeaf22c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d7693a21a716afcbf85b862295bb50d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7f86dd05442e6612faa46448001ab97
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7693a21a716afcbf85b862295bb50d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7f86dd05442e6612faa46448001ab97
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7693a21a716afcbf85b862295bb50d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7f86dd05442e6612faa46448001ab97
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7693a21a716afcbf85b862295bb50d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7f86dd05442e6612faa46448001ab97
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7693a21a716afcbf85b862295bb50d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7f86dd05442e6612faa46448001ab97
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7693a21a716afcbf85b862295bb50d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7f86dd05442e6612faa46448001ab97
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_473777d9c1794849702960ba886c10a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba2566f4bcf666968a19b007a225745c
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fd4d2724f21366174dcc0d4f659c95cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1a888e2572e2939c956dd8a8ab8d6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_9d313334f1be7f4b731da781decfe670(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 80, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_17a21c3a95a86d7edaaf078d9d75a80c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d313334f1be7f4b731da781decfe670
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7b3c87b322c24101810f39d870848d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0ffdd5bf7236aed70be849b15ec5d41
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1c8fe4b5b59b3da7308318058667baf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41445b01980185f9e9f7e623d502fb75
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a91faab716a907cef4740cc24dd1590c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_637f37fde95c5cabe8eb679aee3f3fd5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9770771265029907]], [[0.5021929740905762]], [[0.37103724479675293]], [[1.2088351249694824]], [[0.6041784882545471]], [[-0.14974020421504974]], [[-0.12927401065826416]], [[-0.2439756989479065]], [[-2.9429283142089844]], [[-0.31507694721221924]], [[0.41745054721832275]], [[-0.1007738709449768]], [[0.06134064495563507]], [[0.5260247588157654]], [[0.7378629446029663]], [[-0.7796830534934998]], [[0.9698693752288818]], [[-0.5918015241622925]], [[-2.1311984062194824]], [[0.06964132189750671]], [[1.5712189674377441]], [[-0.6017136573791504]], [[1.0134446620941162]], [[0.2775519788265228]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_32f3784cbf6cae2b6a4768100ecff833(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_38fec47d98b504b6ecabccbec6c5a7c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32f3784cbf6cae2b6a4768100ecff833
    def get_inputs(self):
        return [
            paddle.uniform([22, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_9b599990a1eb100e82a2af14dc9f4494(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9762bb9b9b11fd5e92755e8bbd03651a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b599990a1eb100e82a2af14dc9f4494
    def get_inputs(self):
        return [
            paddle.uniform([22, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7a69fe694299896471829b1b1e6fe180(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 218], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8cdcbf92be15e11fed1c796a03ff9669(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a69fe694299896471829b1b1e6fe180
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_0e8a2e0d31f965248e3339336f64b19e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 872], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 872], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_51e4deb210918e53e47854d2fc324f15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e8a2e0d31f965248e3339336f64b19e
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 872], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f16307de183f24f23b857ac6332a10a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30d93ff451a143c1af4c582ec1de15bd
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d7a15b5c8cbb0593c0dd567247b1faa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_028187fcc580e99a097c9c1e40fe811e
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7e43d3bd20f5359df168f79c87c6c90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b16e12eb901d5e13c56f334912243682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e43d3bd20f5359df168f79c87c6c90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b16e12eb901d5e13c56f334912243682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e43d3bd20f5359df168f79c87c6c90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b16e12eb901d5e13c56f334912243682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e43d3bd20f5359df168f79c87c6c90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b16e12eb901d5e13c56f334912243682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e43d3bd20f5359df168f79c87c6c90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b16e12eb901d5e13c56f334912243682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e43d3bd20f5359df168f79c87c6c90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b16e12eb901d5e13c56f334912243682
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ed757d772991ce7f7e2bf5f22a0fd5fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e64e8082edc6c652cf6ce9093b30699
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.37501490116119385]], [[1.0770702362060547]], [[-1.0405381917953491]], [[0.24033001065254211]], [[-0.028381317853927612]], [[0.4412381052970886]], [[0.840211033821106]], [[0.8207747340202332]], [[0.5067105293273926]], [[-0.17224660515785217]], [[1.2252963781356812]], [[0.9781078100204468]], [[-0.9191286563873291]], [[-0.7753510475158691]], [[-1.7329602241516113]], [[-1.4269700050354004]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_311ed8f3713fbc52e3f347a257c82389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_723f3308f2acc683d28af99dfab14672
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_46bb2565cb498d4814daded00b561779(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2174f9b8cec5e60b28d4a1bad6cdf22
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7b18e13f08ec7601bd78d0fb61bbfbad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7af53569cf6eb3e207d2d6b73a2fbce9
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7b18e13f08ec7601bd78d0fb61bbfbad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7af53569cf6eb3e207d2d6b73a2fbce9
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e3d79641638b3016d46a72068daab85b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b49aa34510522763d72a6c0031adcc7
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1cbcedb3ba854264255d7ff37ed30d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de944bb40bb74a45d9d908ccb174da25
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1cbcedb3ba854264255d7ff37ed30d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de944bb40bb74a45d9d908ccb174da25
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1cbcedb3ba854264255d7ff37ed30d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de944bb40bb74a45d9d908ccb174da25
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1cbcedb3ba854264255d7ff37ed30d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de944bb40bb74a45d9d908ccb174da25
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1cbcedb3ba854264255d7ff37ed30d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de944bb40bb74a45d9d908ccb174da25
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1cbcedb3ba854264255d7ff37ed30d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de944bb40bb74a45d9d908ccb174da25
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8cdcbf92be15e11fed1c796a03ff9669(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a69fe694299896471829b1b1e6fe180
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_51e4deb210918e53e47854d2fc324f15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e8a2e0d31f965248e3339336f64b19e
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 872], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dfd80827fd172cfed6899155de5a4f18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dfd80827fd172cfed6899155de5a4f18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_40945f2cdd1b0139d8e35bd2393935f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_40945f2cdd1b0139d8e35bd2393935f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dfd80827fd172cfed6899155de5a4f18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dfd80827fd172cfed6899155de5a4f18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d003f0b93b2453f6c7d9a275b4232f17(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8400, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8951093895e77fe9b611b7b09984ae08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d003f0b93b2453f6c7d9a275b4232f17
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8400, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0d8bad7df106ca61acc322e3618aaace(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc20a094848a67e308d963ec5296b27d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3f7d1b06be540b4a918d2acf245ad3a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cf47f1fb1d3a0c430b428dfa0bdca51
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_290b9ce10e65281ee9b807e5bbb804de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_637f37fde95c5cabe8eb679aee3f3fd5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.186798557639122]], [[-0.994432806968689]], [[0.8337880373001099]], [[0.8752527236938477]], [[0.39641496539115906]], [[-0.36758700013160706]], [[-0.7982990741729736]], [[0.10533291101455688]], [[-0.7185511589050293]], [[0.19422024488449097]], [[0.10960283875465393]], [[-0.6906976699829102]], [[-1.2756645679473877]], [[-0.9886965751647949]], [[0.31631916761398315]], [[0.06070622801780701]], [[0.2322055697441101]], [[0.06202495098114014]], [[-0.5711102485656738]], [[-0.978243887424469]], [[0.5324698090553284]], [[-0.07336848974227905]], [[1.2363708019256592]], [[-0.17160940170288086]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_619109e79a4fa4d8925f569959669b82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58898b83d91a13f5b58fc4b0af582f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_619109e79a4fa4d8925f569959669b82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58898b83d91a13f5b58fc4b0af582f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_619109e79a4fa4d8925f569959669b82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58898b83d91a13f5b58fc4b0af582f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_619109e79a4fa4d8925f569959669b82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58898b83d91a13f5b58fc4b0af582f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_619109e79a4fa4d8925f569959669b82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58898b83d91a13f5b58fc4b0af582f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_619109e79a4fa4d8925f569959669b82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58898b83d91a13f5b58fc4b0af582f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_16a441208348730a5c7fe93d9bf10fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c9986f333ff64c7e35a9670e33c0a58
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2efd6c58da2618552851b9580eb269ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccaad1c8b27d535fa8f01d6d25382881
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_6b1b54c91382561cdc3d60826d38fff9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 96, 109, 109], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f3a5bf92339103d59cff9be67a641d57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b1b54c91382561cdc3d60826d38fff9
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 109, 109], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_e53598393e75fc6ad7515c759184aff0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 16, 54, 54], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_534ff46b43bc1fcbafcbff631027c5cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e53598393e75fc6ad7515c759184aff0
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_61bb4cb1a1613f24b3bffc6e81e9e58a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 64, 54, 54], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e6991c036cedeca40ebe7712ed6af95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61bb4cb1a1613f24b3bffc6e81e9e58a
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7e6991c036cedeca40ebe7712ed6af95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61bb4cb1a1613f24b3bffc6e81e9e58a
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_534ff46b43bc1fcbafcbff631027c5cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e53598393e75fc6ad7515c759184aff0
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7e6991c036cedeca40ebe7712ed6af95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61bb4cb1a1613f24b3bffc6e81e9e58a
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7e6991c036cedeca40ebe7712ed6af95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61bb4cb1a1613f24b3bffc6e81e9e58a
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_04ea417c3ce80017b485a132c2a5b252(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 32, 54, 54], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e6cee877d5fbc5674a6d6f64975d864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04ea417c3ce80017b485a132c2a5b252
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_1ab1f8954c516438396e33521db8c6aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 128, 54, 54], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e7ed1e91e9702c41f6ca3b474b7b6092(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ab1f8954c516438396e33521db8c6aa
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e7ed1e91e9702c41f6ca3b474b7b6092(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ab1f8954c516438396e33521db8c6aa
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_6bf0f451843a968258bc11b96f3f646a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 32, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_edd3b321b0171c2eb4a767042deab077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bf0f451843a968258bc11b96f3f646a
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_a46f36eb52c10a778562f1229b73449b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 128, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ccb6ac6d281160cb8d27ae0a870e624d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a46f36eb52c10a778562f1229b73449b
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ccb6ac6d281160cb8d27ae0a870e624d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a46f36eb52c10a778562f1229b73449b
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_bfad8353a83944e3174445e6d071268b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 48, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e7182e7ecb6c62273730d650bed96589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfad8353a83944e3174445e6d071268b
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_d2a1cc84d4480b339fd30b24c0fc1c74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 192, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d6d275cc0de9c3b821a3ecb9cd50ef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2a1cc84d4480b339fd30b24c0fc1c74
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7d6d275cc0de9c3b821a3ecb9cd50ef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2a1cc84d4480b339fd30b24c0fc1c74
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e7182e7ecb6c62273730d650bed96589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfad8353a83944e3174445e6d071268b
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7d6d275cc0de9c3b821a3ecb9cd50ef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2a1cc84d4480b339fd30b24c0fc1c74
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7d6d275cc0de9c3b821a3ecb9cd50ef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2a1cc84d4480b339fd30b24c0fc1c74
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_b026c45cea68a1f3bd6e55cc3278389a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 64, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00f44419608a7898d52241ce2b40167b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b026c45cea68a1f3bd6e55cc3278389a
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_bd121365f8f7b7431ddf2cb1c5aa8a6d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 256, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0a533d42a6f07aef6ecd6fd35d9f00b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd121365f8f7b7431ddf2cb1c5aa8a6d
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0a533d42a6f07aef6ecd6fd35d9f00b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd121365f8f7b7431ddf2cb1c5aa8a6d
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_46513ad1e647c8797c735bb00297d69d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 64, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c11d49a1f0028dab51b0bfd4e63748d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46513ad1e647c8797c735bb00297d69d
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_de083583d439bc05227498099a927664(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 256, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c4c911d27d46b3fa690e0b662d761e49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de083583d439bc05227498099a927664
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4c911d27d46b3fa690e0b662d761e49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de083583d439bc05227498099a927664
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_fb9d564f2cd9a0ae193807aee39afd71(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1000, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_72289a220cf38a99f84f329b80ff0e95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb9d564f2cd9a0ae193807aee39afd71
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c48945a461a948ca7b932b45da8ac59d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db9bd9300274c0e9cd7ed6222b6dde73
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9519ce64a4c630a9a2dac537c370bbba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9995740a54c377c85f8688c7f37261f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3bd6f3a37442f1c05c8c690b4daf34b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7db5615c85de2947b7aa067640bd6f20
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.3180536925792694]], [[-0.46659159660339355]], [[0.20483613014221191]], [[0.5790308713912964]], [[0.527435302734375]], [[-0.40371260046958923]], [[0.023192256689071655]], [[0.10091479122638702]], [[0.13100042939186096]], [[0.21230560541152954]], [[-0.36275380849838257]], [[0.049175798892974854]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c48945a461a948ca7b932b45da8ac59d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db9bd9300274c0e9cd7ed6222b6dde73
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7b18e13f08ec7601bd78d0fb61bbfbad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7af53569cf6eb3e207d2d6b73a2fbce9
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e3d79641638b3016d46a72068daab85b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b49aa34510522763d72a6c0031adcc7
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_34c045c64ec0bc8dcb86b9d65407fd73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 23], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c16401aa6fe6fa09decfe667ad8e1840(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34c045c64ec0bc8dcb86b9d65407fd73
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.333017349243164, 0.6476973295211792, 0.31879040598869324, 0.6609146595001221, -1.3607548475265503, -0.7065507173538208, 1.3164575099945068, 0.4826688766479492, -0.40318334102630615, -0.34247010946273804, 1.0047874450683594, -0.24977430701255798, 1.641526460647583, 0.14250433444976807, 0.9402340650558472, -1.2052834033966064, 1.4771676063537598, 1.8669058084487915, 0.09850919246673584, 0.3950137794017792, -0.39424949884414673, -0.05881398916244507, -1.2339973449707031]], dtype='float32').reshape([1, 23]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_1475d51ff7dd4e64471d93af8ab0bbae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 92], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_10ecc49033e6605d2d768e616d26d2d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1475d51ff7dd4e64471d93af8ab0bbae
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0d8bad7df106ca61acc322e3618aaace(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc20a094848a67e308d963ec5296b27d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3f7d1b06be540b4a918d2acf245ad3a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cf47f1fb1d3a0c430b428dfa0bdca51
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7b18e13f08ec7601bd78d0fb61bbfbad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7af53569cf6eb3e207d2d6b73a2fbce9
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_7d41eceafcb7fc8505cd489ee3834125(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d26ed8171faae289a53696e2c54d3fdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d41eceafcb7fc8505cd489ee3834125
    def get_inputs(self):
        return [
            paddle.uniform([22, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_7339b513cd468b8a6dfb4ca0ddd43138(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_211cd09c6fb58e88eb26152a808e790e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7339b513cd468b8a6dfb4ca0ddd43138
    def get_inputs(self):
        return [
            paddle.uniform([22, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_16a441208348730a5c7fe93d9bf10fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c9986f333ff64c7e35a9670e33c0a58
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9519ce64a4c630a9a2dac537c370bbba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9995740a54c377c85f8688c7f37261f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_249ec29597955dc245842917f69b4220(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_411bb89e19b1a9d716ca821e15c913b8
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.4954965114593506]], [[0.9839709401130676]], [[-1.2806291580200195]], [[0.4058416783809662]], [[0.6087191700935364]], [[-0.7118328809738159]], [[-0.2450028955936432]], [[-0.5473366975784302]], [[0.34011638164520264]], [[-0.3276374936103821]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0d8bad7df106ca61acc322e3618aaace(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc20a094848a67e308d963ec5296b27d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_32a3f6080658f7b7936150cafb9690b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_971f74dacec3cc9eb0785c20efb1c69e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.551630973815918]], [[0.3298950493335724]], [[1.7572401762008667]], [[-1.5526847839355469]], [[0.6455285549163818]], [[-0.42094892263412476]], [[-0.6619359850883484]], [[0.7843130230903625]], [[1.1656417846679688]], [[1.6685948371887207]], [[1.353011965751648]], [[-1.6645236015319824]], [[1.5408976078033447]], [[0.7117971777915955]], [[-0.3916815519332886]], [[1.2193541526794434]], [[-0.6401679515838623]], [[0.989652156829834]], [[-0.7580270767211914]], [[0.2924315929412842]], [[0.38859808444976807]], [[-1.4779846668243408]], [[-0.5380164384841919]], [[-2.235255718231201]], [[1.1776700019836426]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9f16307de183f24f23b857ac6332a10a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30d93ff451a143c1af4c582ec1de15bd
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9519ce64a4c630a9a2dac537c370bbba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9995740a54c377c85f8688c7f37261f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_94d5c65c770e1c4d5b5461b80eb72a13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_94d5c65c770e1c4d5b5461b80eb72a13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_146c1a12301cc164716e054f796fb2a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_146c1a12301cc164716e054f796fb2a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_94d5c65c770e1c4d5b5461b80eb72a13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_94d5c65c770e1c4d5b5461b80eb72a13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_90c15181c936c11f09709248d8fea8a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9261, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c84b677a5e93b463571a559b62a42b35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90c15181c936c11f09709248d8fea8a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9261, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_946d13867d5da54fd82c77e06c16d79c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f02610ddc3335384a8f86b4e006c515(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_946d13867d5da54fd82c77e06c16d79c
    def get_inputs(self):
        return [
            paddle.uniform([171, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_97943fd70768ef2da8f96a1f35150ac6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[171, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b884d2aec089167bf45a92b24f68137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97943fd70768ef2da8f96a1f35150ac6
    def get_inputs(self):
        return [
            paddle.uniform([171, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_24c9eea588482f5e9e39b06d0a33c2ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_24c9eea588482f5e9e39b06d0a33c2ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa8e136b60b1f355d10000692c5ae9a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa8e136b60b1f355d10000692c5ae9a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_24c9eea588482f5e9e39b06d0a33c2ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_24c9eea588482f5e9e39b06d0a33c2ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c05a2f4c8d1e6b76f09c61704fe00621(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 7581, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f953bb70aabb05ba70f39e220ee0e1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c05a2f4c8d1e6b76f09c61704fe00621
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 7581, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ecc9897cd90d7318e5c61e659e9a7820(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 312], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c079934a152a151a807d308b45adea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecc9897cd90d7318e5c61e659e9a7820
    def get_inputs(self):
        return [
            paddle.uniform([1, 312], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_376f5a574464d0330ba6dd9544f298b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1248], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1248], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e66ee97a898d08ff3210a0e59714d89c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_376f5a574464d0330ba6dd9544f298b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1248], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fd4d2724f21366174dcc0d4f659c95cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1a888e2572e2939c956dd8a8ab8d6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1e73e0b4e3ba0de55de2c1e175d960d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_236421ce78df60215bd2a7c878c8d700
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f9f96d28b9d9f83faac670a489405119(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97ad36a27c2e642436ff8e15a33cd1a8
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-1.0223602056503296]], [[-0.9295332431793213]], [[-0.2571001648902893]], [[0.19168484210968018]], [[-1.1057995557785034]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1f350a2c88813355bd3670eb9e0c83e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_411bb89e19b1a9d716ca821e15c913b8
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.5863349437713623]], [[-0.11225982010364532]], [[-0.25389039516448975]], [[0.8704276084899902]], [[0.27870675921440125]], [[0.38461053371429443]], [[-0.8996298313140869]], [[0.03190392255783081]], [[-0.1818930208683014]], [[0.9526388049125671]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_a4ab4319490376ac43c7dc640fd80200(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f2c081fe858a99cee7bbd18e12687464(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4ab4319490376ac43c7dc640fd80200
    def get_inputs(self):
        return [
            paddle.to_tensor([[-1.5159015655517578, -0.3717186152935028, -1.2213701009750366, -0.30996429920196533, 1.0166606903076172, -0.9258303046226501, -0.41752856969833374, -0.5561821460723877, -0.15041565895080566, 0.2742302715778351, -0.008862048387527466, 0.11005735397338867, -1.5865366458892822, -0.7527689933776855, -0.03192618489265442, -0.07584860920906067, 0.6713511943817139, 0.6354464888572693, 0.8803795576095581, -0.0368940532207489, 2.149336338043213, 0.14762994647026062, -1.2133398056030273, -0.7538043260574341, -0.4020766615867615, 0.17525345087051392, -0.9040356874465942, 0.19422733783721924, -1.0502251386642456, 1.6136348247528076]], dtype='float32').reshape([1, 30]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_0951099c9eca2809689cbad8e6d88167(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b541fbcde9c09bb057cab215f471002b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0951099c9eca2809689cbad8e6d88167
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1f4d0e4bd69dc9f6325b9da47685660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e46c4d974bba9fa83ba73424a9c938bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1f4d0e4bd69dc9f6325b9da47685660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e46c4d974bba9fa83ba73424a9c938bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1f4d0e4bd69dc9f6325b9da47685660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e46c4d974bba9fa83ba73424a9c938bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1f4d0e4bd69dc9f6325b9da47685660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e46c4d974bba9fa83ba73424a9c938bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1f4d0e4bd69dc9f6325b9da47685660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e46c4d974bba9fa83ba73424a9c938bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1f4d0e4bd69dc9f6325b9da47685660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e46c4d974bba9fa83ba73424a9c938bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1ce50c90f421076c9e5091315ff8999e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d26d447c90282e8cd91c3a291dbfca91
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1a1bd0f3c0f02a9e405d50384e625157(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b70069c758d45982755570bd47cf526
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ae5c423c80a2d03d82a88f66a67efdfe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_41cdeaffda015a8f7e521e30eefda9f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae5c423c80a2d03d82a88f66a67efdfe
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_2df43f53b89b2f31e2a602b54597038f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[145, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e3e4f5b788a6eb39a343c508d25de067(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2df43f53b89b2f31e2a602b54597038f
    def get_inputs(self):
        return [
            paddle.uniform([145, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9519ce64a4c630a9a2dac537c370bbba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9995740a54c377c85f8688c7f37261f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_0d5aafa6fb0a51582dcff66de1c8b3e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a9f8b33f31eeb5820427a0ce830408a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d5aafa6fb0a51582dcff66de1c8b3e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_b9b07e04362e6f99756b188bb855a539(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_90687f580bb419d5ab168bdef6fbb139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b07e04362e6f99756b188bb855a539
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_86a7e2f8da3b0f35122e651b61953818(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[171, 336], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e9d303d5f5a60a808b8f347c9602654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86a7e2f8da3b0f35122e651b61953818
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_031da396d408c031a82ba7263f5ab214(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 32, 100, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 32, 100, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09f07d65f9ae0f5b81544eb869fd2dc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_031da396d408c031a82ba7263f5ab214
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9519ce64a4c630a9a2dac537c370bbba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9995740a54c377c85f8688c7f37261f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2efd6c58da2618552851b9580eb269ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccaad1c8b27d535fa8f01d6d25382881
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_16a441208348730a5c7fe93d9bf10fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c9986f333ff64c7e35a9670e33c0a58
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c48945a461a948ca7b932b45da8ac59d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db9bd9300274c0e9cd7ed6222b6dde73
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_90687f580bb419d5ab168bdef6fbb139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b07e04362e6f99756b188bb855a539
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7e9d303d5f5a60a808b8f347c9602654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86a7e2f8da3b0f35122e651b61953818
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_16a441208348730a5c7fe93d9bf10fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c9986f333ff64c7e35a9670e33c0a58
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_4735be63a4619577c68fc470e1a3f8b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 6, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a0dfab731f18d332f05da2aed2c033c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4735be63a4619577c68fc470e1a3f8b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a923962b0ef84fad1bd205b706f93587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec36b70774ed7c07b274b3f151805f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a923962b0ef84fad1bd205b706f93587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec36b70774ed7c07b274b3f151805f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a923962b0ef84fad1bd205b706f93587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec36b70774ed7c07b274b3f151805f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a923962b0ef84fad1bd205b706f93587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec36b70774ed7c07b274b3f151805f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a923962b0ef84fad1bd205b706f93587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec36b70774ed7c07b274b3f151805f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a923962b0ef84fad1bd205b706f93587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec36b70774ed7c07b274b3f151805f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_70cb339b3f7a24cedc26cc1de6728434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aaf550957e26b3fa1ff256401484b75d
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_612405d9a4df6d531634c00f8cb5d2ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94b687023f1365f8a555ab117618d184
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.4779714345932007]], [[-1.4063608646392822]], [[0.44086337089538574]], [[0.41031360626220703]], [[-1.464568853378296]], [[-0.34897446632385254]], [[0.02108362317085266]], [[0.4990442991256714]], [[0.6028696298599243]], [[0.5234121084213257]], [[1.1683506965637207]], [[-0.4526069760322571]], [[0.5388213396072388]], [[1.0437719821929932]]]], dtype='float32').reshape([1, 14, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_287adb5cadba02410e154a6ccedb126e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b222b3985c2922e29e86fce7d504551b
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_62c100f5508a3a8c6df85d10c67b4d24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e64e8082edc6c652cf6ce9093b30699
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.2339308261871338]], [[0.4682016372680664]], [[0.5151702165603638]], [[2.317068099975586]], [[-0.4595473110675812]], [[0.9086174964904785]], [[0.6631268262863159]], [[-0.21437422931194305]], [[-1.4097681045532227]], [[-0.027055412530899048]], [[-1.5439503192901611]], [[-1.1007261276245117]], [[0.05410495400428772]], [[-0.8391332030296326]], [[-0.2884824872016907]], [[-0.4638057351112366]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_311ed8f3713fbc52e3f347a257c82389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_723f3308f2acc683d28af99dfab14672
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_4f34ab314837124d285f1fcecb43a6d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5bf6b159b7e21e670f19963476d3d8d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f34ab314837124d285f1fcecb43a6d8
    def get_inputs(self):
        return [
            paddle.uniform([10, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_4b5ddc3dace7d894ea2db264f5a9f8ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1cfc899c6fe011acb0c1b438121c7599(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5ddc3dace7d894ea2db264f5a9f8ca
    def get_inputs(self):
        return [
            paddle.uniform([10, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c4008a7a702420ae8093bf99bb70c5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f459f1d551d0eaa20ac62332b1bc1ddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_b30dcdf420a18c2832f805a6bb2b90cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_90567f414261a536afe14f228d523759(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b30dcdf420a18c2832f805a6bb2b90cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_62126afc9716bf86dafa961da8791c77(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 300, 300], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6e4455f7f224af1097aedc62e67f7aae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62126afc9716bf86dafa961da8791c77
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6e4455f7f224af1097aedc62e67f7aae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62126afc9716bf86dafa961da8791c77
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_b57ab302946a8f030b90a079cab1ced9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 150, 150], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b8089b8337197284cfdf1d2c8f7e0a38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b57ab302946a8f030b90a079cab1ced9
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b8089b8337197284cfdf1d2c8f7e0a38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b57ab302946a8f030b90a079cab1ced9
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_bb39117eafb0508f7bdaa6ade2f0f388(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 75, 75], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5bed5c4ec95f13cee6893963c194f2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb39117eafb0508f7bdaa6ade2f0f388
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5bed5c4ec95f13cee6893963c194f2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb39117eafb0508f7bdaa6ade2f0f388
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5bed5c4ec95f13cee6893963c194f2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb39117eafb0508f7bdaa6ade2f0f388
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_a80807c9c9952f791c20eb852b786bfa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 38, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cfe73c339d5dbeba4bed84447d1705b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a80807c9c9952f791c20eb852b786bfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cfe73c339d5dbeba4bed84447d1705b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a80807c9c9952f791c20eb852b786bfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cfe73c339d5dbeba4bed84447d1705b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a80807c9c9952f791c20eb852b786bfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_ada08d565accc2638bc7c66e98964871(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 19, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7f5d7749fb2efa8af2764a9ed2bbef06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ada08d565accc2638bc7c66e98964871
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7f5d7749fb2efa8af2764a9ed2bbef06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ada08d565accc2638bc7c66e98964871
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7f5d7749fb2efa8af2764a9ed2bbef06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ada08d565accc2638bc7c66e98964871
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_a1f6888212b8aa9407c24ded8434b494(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 19, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a358a1684be553b727e2a8d05cca48e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1f6888212b8aa9407c24ded8434b494
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a358a1684be553b727e2a8d05cca48e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1f6888212b8aa9407c24ded8434b494
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_d6476e4860c7a9d761ffc7a47fca8a3e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 19, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_720db9e5e4c785934a3ecc2870bd72ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6476e4860c7a9d761ffc7a47fca8a3e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_a9596474c68d715542b6d30c6022f66c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 10, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8af03c6cca9f9e7a7b18f2799858e7aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9596474c68d715542b6d30c6022f66c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_2dddb66130d72a2ced8f8a191706ec55(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 10, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b5025e2a38b36b79f15020f263380fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2dddb66130d72a2ced8f8a191706ec55
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_05fb0b6c4b77bdf9d37435b6af586af6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 5, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_db43a2e8ca848f53446b5ed1da669c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05fb0b6c4b77bdf9d37435b6af586af6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_535d990718c37b890d283ff3f86dec4b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 5, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_efa215d930a8532546a614c7926328d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_535d990718c37b890d283ff3f86dec4b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 5, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_96eb11dd7f4e14320ae5a6b5dc283f42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_34ab87ff7930bf39c9f6c2694df2734f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96eb11dd7f4e14320ae5a6b5dc283f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_3b035c70ad9e1a7a509d2366ced7f9ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c7abb62e34d5986344aa8a2b76aadcd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b035c70ad9e1a7a509d2366ced7f9ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_826ff0fce8ffd4f998f89b2ddc057265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b965144fdb6428a939758f26b8266b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
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


class TestPrimitiveOp_485de7b1d51e177a1ccc98afa17f2afd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd9731f87a86b54aba40f818340e1c5f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.000000013351432e-10], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_720558464403804977c7463e452e36c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28e2137aee26b5ebeb884e5eee400c3c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.7418999671936035]], [[-0.6221029758453369]], [[-1.4528839588165283]], [[-1.3356382846832275]], [[2.218531847000122]], [[-0.10492527484893799]], [[-0.1132655143737793]], [[0.6670564413070679]], [[0.2132931351661682]], [[1.139336347579956]], [[-0.38198378682136536]], [[0.5378648042678833]], [[-1.1248897314071655]], [[0.5455056428909302]], [[-0.9092668294906616]], [[0.16010060906410217]], [[-0.6187860369682312]], [[-0.30791616439819336]], [[0.037060633301734924]], [[1.3559787273406982]], [[-0.7795363068580627]], [[0.5538970828056335]], [[-0.21427369117736816]], [[0.2776618003845215]], [[-0.2322857826948166]], [[0.7336132526397705]], [[1.0730092525482178]], [[-0.9537456631660461]], [[0.5654987096786499]], [[0.1285128891468048]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5391c92b5568fb3cb2312949c886f52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899a334f90af93f7c1c99e41cb1ebfbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac5b60e32655f106e5cb57a1ce901423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c6c010d8fe72871bd214119eeaf22c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e3ae15e3947441f0b2eccb4335fcefde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_832094585afa456c071fc94e07acf7f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_28e6c6739b1f6d3fa0d07d0e895e480f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beb2866ad646a4731fb9007aee1a16dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9d5fd8fd3a119636747083813b00fddf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40aeb0fd2b2dd8fe956835b6bdd3594d
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_311ed8f3713fbc52e3f347a257c82389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_723f3308f2acc683d28af99dfab14672
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_826ff0fce8ffd4f998f89b2ddc057265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b965144fdb6428a939758f26b8266b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_473777d9c1794849702960ba886c10a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba2566f4bcf666968a19b007a225745c
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fd4d2724f21366174dcc0d4f659c95cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1a888e2572e2939c956dd8a8ab8d6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_54150b312330195002bbf106716197a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 92, 140], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f26e37ae0d3f615e3d7fd460489829e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54150b312330195002bbf106716197a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]




if __name__ == '__main__':
    unittest.main()