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



class PrimitiveOp_0afd010202d1a969c7778d34d9471fca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 672, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0eb2656ff4e8b1636af2d800b2584cf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0afd010202d1a969c7778d34d9471fca
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_af906769dc4eff5d91efc4def85f9ccb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 28, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02695ceebbec13ddb0b55054035cb687(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af906769dc4eff5d91efc4def85f9ccb
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3a9a7e54cdd157ffb07d233b950f63c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 96, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9ffe79a9a87d127ba1212d6e399c5160(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a9a7e54cdd157ffb07d233b950f63c5
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d792f3382c40c9d072ada19cb0dc95a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 4, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_79ae24e329341ecc32aabf4bd81ea2ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d792f3382c40c9d072ada19cb0dc95a2
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4759cf7ffd05763aeb77b9ec2d6b1498(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 240, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e201ef2333ec83c161855b0f73dc1095(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4759cf7ffd05763aeb77b9ec2d6b1498
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_35a0dba4b6e1d7e1599db78756b0de7f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 10, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_68df4b68a105f43a22a123a80999b08c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35a0dba4b6e1d7e1599db78756b0de7f
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0eb2656ff4e8b1636af2d800b2584cf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0afd010202d1a969c7778d34d9471fca
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_02695ceebbec13ddb0b55054035cb687(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af906769dc4eff5d91efc4def85f9ccb
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_58e76a6de146aaa12ef03af463255f1f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 240, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f7de91ed768b813eb8dc123d11dbaa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58e76a6de146aaa12ef03af463255f1f
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b3935af871564d57467b02e90e6e3e07(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 10, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c607acda67ffc784ba25478d38bd9ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3935af871564d57467b02e90e6e3e07
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_86b7a608e0926c4fc21de65c59e41eda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 672, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_44771b155d17ae6d46f2b0ac94b9a973(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86b7a608e0926c4fc21de65c59e41eda
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_02695ceebbec13ddb0b55054035cb687(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af906769dc4eff5d91efc4def85f9ccb
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1ff61dc889eafd7a72ce44eccd685c57(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8132f0d49977d7a6fb34344f37d89a8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ff61dc889eafd7a72ce44eccd685c57
    def get_inputs(self):
        return [
            paddle.uniform([11, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f71db8c0e45b892fd28f32c3f9c4690c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 672, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_76a3215101763d8a9cb6768b3409a532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f71db8c0e45b892fd28f32c3f9c4690c
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_399eaf9c137f73f40abd7182e5336b93(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 28, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e72d0c8f90ae8519474801a0b3e7b8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_399eaf9c137f73f40abd7182e5336b93
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_44771b155d17ae6d46f2b0ac94b9a973(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86b7a608e0926c4fc21de65c59e41eda
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_02695ceebbec13ddb0b55054035cb687(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af906769dc4eff5d91efc4def85f9ccb
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8c607acda67ffc784ba25478d38bd9ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3935af871564d57467b02e90e6e3e07
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6334af5f5264775b503ecf1ab93b01f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 672, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc306c3474be372f7f6e5a9b181907f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6334af5f5264775b503ecf1ab93b01f3
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9e72d0c8f90ae8519474801a0b3e7b8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_399eaf9c137f73f40abd7182e5336b93
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_afa638378a69ac6c019dcab2895bab48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 32, 112, 112], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d0bffa97cb8ecec3b483b10f21f8a86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afa638378a69ac6c019dcab2895bab48
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_81ab812b84401e2b4638119456bbf874(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 8, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a18f48a709f6fdf17b15f07eca0d687(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81ab812b84401e2b4638119456bbf874
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e201ef2333ec83c161855b0f73dc1095(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4759cf7ffd05763aeb77b9ec2d6b1498
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_68df4b68a105f43a22a123a80999b08c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35a0dba4b6e1d7e1599db78756b0de7f
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5af752dd6a9ee26cd193f1d943647b13(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 144, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97f3e2107594699cd9636122cd7ae742(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5af752dd6a9ee26cd193f1d943647b13
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8dc6c92c6ede6df5fc389555f2dcf7bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 6, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f24b80adaf751c9b72d830956afbca01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dc6c92c6ede6df5fc389555f2dcf7bc
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_324ad1556e71000250b70ba55192c9da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 32, 112, 112], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a10216a75a981a0ad2685290cc0d2b45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_324ad1556e71000250b70ba55192c9da
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6bb073c3483a034f3d0eae7c7cf3b866(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 8, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f7dafc5d6a12417534835e5cc0f0af1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bb073c3483a034f3d0eae7c7cf3b866
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5e766484693ad567d463512b34d68a2d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1152, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c65fd43fe0562bad3a7c26f206beb51b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e766484693ad567d463512b34d68a2d
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fa3fe3c466a7887b8262d47c1401f02f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ff0e4a6ff38c4a9cba9636dbf1000729(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa3fe3c466a7887b8262d47c1401f02f
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_45153586e9b8ff7f9bbbfacef60a4216(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 96, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c4c5fde8a189b470cbc8b927e1ceb90c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45153586e9b8ff7f9bbbfacef60a4216
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_099ca2d1adf4b65f61e6cae6439da22e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 4, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b437a082808dfac3b1d17d61d8f0bb39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_099ca2d1adf4b65f61e6cae6439da22e
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0a724e2d6b5a1a2fcc380325f9355392(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 144, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b31d0cb3f7cfb48d7d96f6b0c7ac4fcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a724e2d6b5a1a2fcc380325f9355392
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fffc560b15a66de5ed5ead4231ed6321(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 6, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ebdc12cc57a50def33815a46ddb6cf90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fffc560b15a66de5ed5ead4231ed6321
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c65fd43fe0562bad3a7c26f206beb51b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e766484693ad567d463512b34d68a2d
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ff0e4a6ff38c4a9cba9636dbf1000729(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa3fe3c466a7887b8262d47c1401f02f
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_aa181ead4113466cf1bda7ae0d3f76ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1152, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_35e5a6ab3882a8e3eea4af564943838d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa181ead4113466cf1bda7ae0d3f76ef
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d16d2cd28c5c7e54498e3af79579e2c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3c1e797f1a70221bcaab12dcf036c5db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d16d2cd28c5c7e54498e3af79579e2c9
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3c443407a93339000870365ac6c41dd1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 480, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ecfbc2912230918ebd027ea9b416ca2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c443407a93339000870365ac6c41dd1
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8132f0d49977d7a6fb34344f37d89a8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ff61dc889eafd7a72ce44eccd685c57
    def get_inputs(self):
        return [
            paddle.uniform([11, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_14091d51bdef1d9f7aa7049d77870f86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 480, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55250e788043edff0e87814d48f179df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14091d51bdef1d9f7aa7049d77870f86
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_95ce95e3d0060a549572c4f88577832f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d615ceadb35e51b3cd9068d3a9f92b94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95ce95e3d0060a549572c4f88577832f
    def get_inputs(self):
        return [
            paddle.uniform([43, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0d0bffa97cb8ecec3b483b10f21f8a86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afa638378a69ac6c019dcab2895bab48
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7a18f48a709f6fdf17b15f07eca0d687(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81ab812b84401e2b4638119456bbf874
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8c607acda67ffc784ba25478d38bd9ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3935af871564d57467b02e90e6e3e07
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_44063bcb67fc26548398d36f94dec837(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 240, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6bf91bd4892f3b6ca3bb59a25c6d8cbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44063bcb67fc26548398d36f94dec837
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_68df4b68a105f43a22a123a80999b08c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35a0dba4b6e1d7e1599db78756b0de7f
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b31d0cb3f7cfb48d7d96f6b0c7ac4fcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a724e2d6b5a1a2fcc380325f9355392
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ebdc12cc57a50def33815a46ddb6cf90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fffc560b15a66de5ed5ead4231ed6321
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9e72d0c8f90ae8519474801a0b3e7b8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_399eaf9c137f73f40abd7182e5336b93
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3c1e797f1a70221bcaab12dcf036c5db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d16d2cd28c5c7e54498e3af79579e2c9
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ce05efa0a6f1388b821fa8b80b31571d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 144, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cea567a4d9191ab6e55dad5ffbfae40a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce05efa0a6f1388b821fa8b80b31571d
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ebdc12cc57a50def33815a46ddb6cf90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fffc560b15a66de5ed5ead4231ed6321
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9e72d0c8f90ae8519474801a0b3e7b8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_399eaf9c137f73f40abd7182e5336b93
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f24b80adaf751c9b72d830956afbca01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dc6c92c6ede6df5fc389555f2dcf7bc
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d3003c2d5d093beb262bfe92f84514d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 240, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_075cbb2a2fe4fca20577440db296ce3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3003c2d5d093beb262bfe92f84514d6
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8c607acda67ffc784ba25478d38bd9ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3935af871564d57467b02e90e6e3e07
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a10216a75a981a0ad2685290cc0d2b45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_324ad1556e71000250b70ba55192c9da
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f7dafc5d6a12417534835e5cc0f0af1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bb073c3483a034f3d0eae7c7cf3b866
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_97f3e2107594699cd9636122cd7ae742(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5af752dd6a9ee26cd193f1d943647b13
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f24b80adaf751c9b72d830956afbca01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dc6c92c6ede6df5fc389555f2dcf7bc
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_55250e788043edff0e87814d48f179df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14091d51bdef1d9f7aa7049d77870f86
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d615ceadb35e51b3cd9068d3a9f92b94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95ce95e3d0060a549572c4f88577832f
    def get_inputs(self):
        return [
            paddle.uniform([43, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9ffe79a9a87d127ba1212d6e399c5160(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a9a7e54cdd157ffb07d233b950f63c5
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_79ae24e329341ecc32aabf4bd81ea2ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d792f3382c40c9d072ada19cb0dc95a2
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d6fa4331405c9eda15968003b1789852(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 144, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_12d0f6493c8390b23459aa710e322b88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6fa4331405c9eda15968003b1789852
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f24b80adaf751c9b72d830956afbca01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dc6c92c6ede6df5fc389555f2dcf7bc
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cea567a4d9191ab6e55dad5ffbfae40a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce05efa0a6f1388b821fa8b80b31571d
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ebdc12cc57a50def33815a46ddb6cf90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fffc560b15a66de5ed5ead4231ed6321
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c4c5fde8a189b470cbc8b927e1ceb90c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45153586e9b8ff7f9bbbfacef60a4216
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b437a082808dfac3b1d17d61d8f0bb39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_099ca2d1adf4b65f61e6cae6439da22e
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6bf91bd4892f3b6ca3bb59a25c6d8cbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44063bcb67fc26548398d36f94dec837
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_68df4b68a105f43a22a123a80999b08c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35a0dba4b6e1d7e1599db78756b0de7f
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()