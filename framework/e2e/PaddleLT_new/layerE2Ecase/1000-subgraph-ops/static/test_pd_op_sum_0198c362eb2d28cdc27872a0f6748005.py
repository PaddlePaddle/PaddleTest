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



class PrimitiveOp_63face4a3992a135524ed8e45fd0fc4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 576, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dcd894b3ff471b8f6e50d861e04b9f0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63face4a3992a135524ed8e45fd0fc4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_0e2dccc70d629c3ad7715800adda3ab8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 136, 136], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c98498f8d4b40541a42f8111e500ddfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2dccc70d629c3ad7715800adda3ab8
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 136, 136], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_6c53897818ae9630e173eb9e7385db1a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 21824, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5eb8725d8196b0bbd5e3c903a1bd47a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c53897818ae9630e173eb9e7385db1a
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5eb8725d8196b0bbd5e3c903a1bd47a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c53897818ae9630e173eb9e7385db1a
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0451eeb7efec99cf33667bf194111aa6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3aa2ef2ce58f255f901a81c034c8e135(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0451eeb7efec99cf33667bf194111aa6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.011248950846493244, 0.015533224679529667]], [[0.056182388216257095, 0.06118789315223694]], [[0.07811105996370316, 0.13334205746650696]], [[0.033715955913066864, 0.000653924245852977]], [[0.15784281492233276, 0.00011422202078392729]], [[0.08321346342563629, 0.38469576835632324]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_648ac687d87792387cadff494c4f718d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0451eeb7efec99cf33667bf194111aa6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.22401949763298035, 0.05460046976804733]], [[0.21837912499904633, 0.06345872581005096]], [[0.5931506156921387, 0.9614284634590149]], [[0.00036927530891261995, 0.18518532812595367]], [[0.024074211716651917, 0.48831549286842346]], [[0.005147542804479599, 0.04141313582658768]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5b8a87c84c7f7ac866ff73ee3f9861cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 100, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_78a45f5f17bbbaa50547c6a470768d08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b8a87c84c7f7ac866ff73ee3f9861cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_78a45f5f17bbbaa50547c6a470768d08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b8a87c84c7f7ac866ff73ee3f9861cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c2365f9505ee300638e59dcb407eb5d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 100, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4cc85c48c943528f22137ac727281009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2365f9505ee300638e59dcb407eb5d8
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1ba897b1a7eb20fe914cff1ff8626b74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 2, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc30fe871f7b4b8ccb4a3ca9fb70424c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ba897b1a7eb20fe914cff1ff8626b74
    def get_inputs(self):
        return [
            paddle.uniform([100, 2, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_86fb362d08732a5abec6ff4a66865729(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_606c9d7fc3c8fa940767dbd8a9957e02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86fb362d08732a5abec6ff4a66865729
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_606c9d7fc3c8fa940767dbd8a9957e02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86fb362d08732a5abec6ff4a66865729
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_41f5267601ee7d7602cdf85fe49d1c19(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1025, 1025], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8563ea9ef4b0ceae721255044a63dc61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41f5267601ee7d7602cdf85fe49d1c19
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9ca35e222902e313f9424c2b95a74c37(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f58dd73b68ca609f42da846670a91054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ca35e222902e313f9424c2b95a74c37
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_50565d800ea894212a0f12151fe22f1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1542, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_58296bf0aa65b1abce226b588ff27ba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50565d800ea894212a0f12151fe22f1c
    def get_inputs(self):
        return [
            paddle.uniform([1542, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_5eb2cb6f70280b8c80dee581478be8a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1542, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_220afd1b121f0ce0f07f5066bca772b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5eb2cb6f70280b8c80dee581478be8a2
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_79f5fb81740e3fe65fa2062f523eb1e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1542, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_35466775dfe585294d5ad8089d0db9d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79f5fb81740e3fe65fa2062f523eb1e3
    def get_inputs(self):
        return [
            paddle.uniform([1542, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_220afd1b121f0ce0f07f5066bca772b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5eb2cb6f70280b8c80dee581478be8a2
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_e00148bf9f8335400b542cc1411a67e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e266f1b99a14df663b7692725e3fceea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e00148bf9f8335400b542cc1411a67e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_66011e0b809ba26203da5d746b2b863b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2361, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_31cc140d3329a3c12136741ae765efc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66011e0b809ba26203da5d746b2b863b
    def get_inputs(self):
        return [
            paddle.uniform([2361, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_81d6d9214e97da72c32218205280ff15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2361, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9dbd3412b1bc7f9f3d93068bc189bafd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81d6d9214e97da72c32218205280ff15
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_8be2e85607c3c81b7280377203908e16(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2361, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92c5eb62b7b177ecc11675e877a90f88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8be2e85607c3c81b7280377203908e16
    def get_inputs(self):
        return [
            paddle.uniform([2361, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9dbd3412b1bc7f9f3d93068bc189bafd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81d6d9214e97da72c32218205280ff15
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_1e03e78c643c36f9ac024f3cb15c3bc4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e587a63b4d8c8f243e9de2dc259d1bac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e03e78c643c36f9ac024f3cb15c3bc4
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_3630cee931cd7b85632f4fbb057921d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2a9f386fea26638ad62b05b715610be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3630cee931cd7b85632f4fbb057921d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b2a9f386fea26638ad62b05b715610be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3630cee931cd7b85632f4fbb057921d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_fe5cac74bbaa59835f4f7033082872b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e57679d89c2458b6a99d64793500f07a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe5cac74bbaa59835f4f7033082872b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_fe0127b6aef1460ad59f4626983cd1ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 2304, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_30b81bd2f5cedce278b5a3cc5b47b81c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe0127b6aef1460ad59f4626983cd1ab
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_30b81bd2f5cedce278b5a3cc5b47b81c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe0127b6aef1460ad59f4626983cd1ab
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_89d93a4298ffafba0d72da3f76a95cc3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 3136, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c92256012b7ca583c1dd3d672cc5db7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89d93a4298ffafba0d72da3f76a95cc3
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c92256012b7ca583c1dd3d672cc5db7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89d93a4298ffafba0d72da3f76a95cc3
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ac7a96a1a21905a7f6c975a73a006eec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 120, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c2b2419f164d0551642968e50d047f68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac7a96a1a21905a7f6c975a73a006eec
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 120, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_11b890bb65900f90f2880f27a9556e64(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 3136, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bfc42a3cad9f0d1754fcce21d3e141fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11b890bb65900f90f2880f27a9556e64
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bfc42a3cad9f0d1754fcce21d3e141fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11b890bb65900f90f2880f27a9556e64
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0ec9cd30550a4fbd82462c3858b2cbc7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 64, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_029d505d74c71e5a85962f6ca67758d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ec9cd30550a4fbd82462c3858b2cbc7
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_029d505d74c71e5a85962f6ca67758d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ec9cd30550a4fbd82462c3858b2cbc7
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ad5b49f096c37ec3bca9ec38824a3b77(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 128, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_61b86c5aa3ce912a964d107c206ac35a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad5b49f096c37ec3bca9ec38824a3b77
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_61b86c5aa3ce912a964d107c206ac35a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad5b49f096c37ec3bca9ec38824a3b77
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_29a3ed6ebd61206e3d2cd6f6c680b8dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 256, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_da7163a9e58d18300c1d10293015de31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29a3ed6ebd61206e3d2cd6f6c680b8dc
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_da7163a9e58d18300c1d10293015de31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29a3ed6ebd61206e3d2cd6f6c680b8dc
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7b7fdaa99413d75d967428fc12f38563(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 512, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3d63f530a5d49615277ac5ab96283351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b7fdaa99413d75d967428fc12f38563
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3d63f530a5d49615277ac5ab96283351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b7fdaa99413d75d967428fc12f38563
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d5092d496a9f82cc0016c5dd542d3883(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8152c57d3fcf77df607db30fb0e32ef0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5092d496a9f82cc0016c5dd542d3883
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8152c57d3fcf77df607db30fb0e32ef0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5092d496a9f82cc0016c5dd542d3883
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f2661a5de0ea4690c14156c186869a96(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e0af0076e242d2f637a47270a4e6e54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f2661a5de0ea4690c14156c186869a96
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_8ffeb1b54e977edd3aee44a810d2689b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 197, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fb4b7709fdfe4f99649eb4e794112274(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ffeb1b54e977edd3aee44a810d2689b
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fb4b7709fdfe4f99649eb4e794112274(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ffeb1b54e977edd3aee44a810d2689b
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_40044a44c485f7f301ab3d28ea8f6a35(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 197, 197], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fb1f36b6c4d601991f38c20a0a02cf23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40044a44c485f7f301ab3d28ea8f6a35
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 197], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_dac9c99830c5b556862b0366080ff63d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 784, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab15839faa20888f2ea55a2ab6d6576c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dac9c99830c5b556862b0366080ff63d
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ab15839faa20888f2ea55a2ab6d6576c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dac9c99830c5b556862b0366080ff63d
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_bd60ccf92575e1c65b4427f0c7679788(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8816], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b1d868a7db2674c9778226487db6164(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd60ccf92575e1c65b4427f0c7679788
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_1cb515b191ea43982d737086bee3150a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6aec482badb9de8039d271bd03054ec0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1cb515b191ea43982d737086bee3150a
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6c3bf086e0a99c58554b1441f88527af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1cc2eaaefc6632e6bddb82eff1b35b72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c3bf086e0a99c58554b1441f88527af
    def get_inputs(self):
        return [
            paddle.to_tensor([2.594792127609253, 1.27143394947052, 1.972851276397705, 3.492600917816162, 3.574134349822998, 2.861876964569092], dtype='float32').reshape([6]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a13a99c0a281f81ea3ece857939576f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 4, 16, 49, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d2bd426e7549cb03c9ad4ddc4c26eab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a13a99c0a281f81ea3ece857939576f5
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0ae17d630b00a9a62b9882afd471334e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[47, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_33e76d6bf3ef8b0760039d90ab4f7416(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ae17d630b00a9a62b9882afd471334e
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_a8e70b4dea7220bf0ff89d8875ad2ebb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_effe0f3632e15153d8b8ffb80dfbe655(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8e70b4dea7220bf0ff89d8875ad2ebb
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_00d7d9283dd4c29fb1dd1d6d8613fda8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 32, 16, 49, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a74c7347b11a2030401682c8763dbff5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00d7d9283dd4c29fb1dd1d6d8613fda8
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a8f2ed709d5ede07eba99d8ceed79343(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 8, 16, 49, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f95ca53409b265af3d50ba9f34f5ba9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8f2ed709d5ede07eba99d8ceed79343
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_323d19446faca91132f2906871d32852(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15200], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_061e2f6c77ec0d3beb825a56582b854d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_323d19446faca91132f2906871d32852
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_7faedab13c0021f60b3234f662e17f47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 168, 168], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_585f7595fe7125e6a8b8b8c04e24dd2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7faedab13c0021f60b3234f662e17f47
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 168, 168], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_a84549388426dd411deb264c608cf2e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_54aee6e1254df9858f88252f2d4d884b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84549388426dd411deb264c608cf2e5
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_54aee6e1254df9858f88252f2d4d884b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84549388426dd411deb264c608cf2e5
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c9d82becd1bcbd180ba17c48dca58ecd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_386f3db1da08f46b5957227b5ac0d4b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d82becd1bcbd180ba17c48dca58ecd
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_386f3db1da08f46b5957227b5ac0d4b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d82becd1bcbd180ba17c48dca58ecd
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d2eb1c244dc99394205841c096b3a4f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2725000fbdb5667bdea3c330f5dcd54b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2eb1c244dc99394205841c096b3a4f2
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_05e072ee68558971315f58207778356f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 4, 50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a1b95d2173950c9fb22673437d3de87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05e072ee68558971315f58207778356f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_93a12b4938730db7937bd7f76e97aa52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 640, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c0e999728d37c11be93421a84030614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93a12b4938730db7937bd7f76e97aa52
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1c0e999728d37c11be93421a84030614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93a12b4938730db7937bd7f76e97aa52
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_bff9dc55d64c04caee29a8fd5c5849d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 640, 640], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afd50fe3b31765836c19887dcc5afaf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bff9dc55d64c04caee29a8fd5c5849d1
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 640], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f749254b81779169df24a356dd1d2bca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 2, 16, 9, 112, 112], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9ca73c2ba8706421b3004eec38ba223f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f749254b81779169df24a356dd1d2bca
    def get_inputs(self):
        return [
            paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1e9f3ba0e7ec34a9ad3d80aa393d413f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 104, 104], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6720fb06c15591d20eb505a38d17ec68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e9f3ba0e7ec34a9ad3d80aa393d413f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 104, 104], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_54aee6e1254df9858f88252f2d4d884b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84549388426dd411deb264c608cf2e5
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_54aee6e1254df9858f88252f2d4d884b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84549388426dd411deb264c608cf2e5
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_47a42dcf301b1b0acd0b4c1a47aa502b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 576, 13, 13], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ad17da946b01addeb934b29902d5c8c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47a42dcf301b1b0acd0b4c1a47aa502b
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_e62e0d3972589748c949d185b1fbb896(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 2, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e27ad823f8cab2fda79c595a1c3efef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62e0d3972589748c949d185b1fbb896
    def get_inputs(self):
        return [
            paddle.uniform([300, 2, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4e44634637cea890677466d8c40c8366(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 52, 52], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2029c1a15981a6edf56eb1e5b399392b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e44634637cea890677466d8c40c8366
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_ea44b40afddc83ea56bfc40d4389c2d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d10fdb2faad851f3e2ba7b6d78b0382(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea44b40afddc83ea56bfc40d4389c2d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6c65106ce0b50ac5f74b2ac7cb23fd5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 4, 100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1171bf654ec0d145a4eabb943a644cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c65106ce0b50ac5f74b2ac7cb23fd5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a476fe34594a261e3ac69f50d57148f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21, 16384], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d51ed03e63381e287d7ebafafcee7d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a476fe34594a261e3ac69f50d57148f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 16384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c92256012b7ca583c1dd3d672cc5db7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89d93a4298ffafba0d72da3f76a95cc3
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c92256012b7ca583c1dd3d672cc5db7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89d93a4298ffafba0d72da3f76a95cc3
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ff77f64f01a7106b76367449ea20478a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 200, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_171f685c0dae292b4dc1b88343cc66dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff77f64f01a7106b76367449ea20478a
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_171f685c0dae292b4dc1b88343cc66dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff77f64f01a7106b76367449ea20478a
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_cff73762a0f983bad5687691e5dc0822(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 200, 200], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a9f058069d00495954ec134906da2e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cff73762a0f983bad5687691e5dc0822
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1044f941ffc5db497c999a20857202cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e328f940e2e2c7b893f8ca9712a8d64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1044f941ffc5db497c999a20857202cf
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2e328f940e2e2c7b893f8ca9712a8d64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1044f941ffc5db497c999a20857202cf
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_649e51e8c1d385436750b626ae205186(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8abc75ab80e162b15add84f6518060e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_649e51e8c1d385436750b626ae205186
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_6c7e8f4b4b6bdc423e3bb8b94beba4af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e93975169efd752249541e949ad981f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c7e8f4b4b6bdc423e3bb8b94beba4af
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0acf712efb0207bb7443f2f9edcb5940(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d24402dbdb89f4f65804ead474b69947(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0acf712efb0207bb7443f2f9edcb5940
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d24402dbdb89f4f65804ead474b69947(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0acf712efb0207bb7443f2f9edcb5940
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f7bd4a9ca444d9cad3ee3d44d96712ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b30f0345f149b1a0f1e457ce29f65fa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7bd4a9ca444d9cad3ee3d44d96712ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_76d2cc8e16b896bbb0ec329ed67f2acf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e566b295b419f94cdd85243ed45ded9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76d2cc8e16b896bbb0ec329ed67f2acf
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_ea7831c0d6050d1b61b4dd4452922e1d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8dcda1ab3dd1045ac9106ef5c4e5841d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea7831c0d6050d1b61b4dd4452922e1d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.17037688195705414, 0.1606130450963974, 0.1035061702132225, 0.0991436243057251], dtype='float32').reshape([4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_05df947d6acd8be87e09b49d9cb4f6f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21824, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_32f628bcdc9ddc91df37c53fd30328f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05df947d6acd8be87e09b49d9cb4f6f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_9f2389de34a4bfaca378ee750e68aa13(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09d350bd367974f4fe007ab94f2e3aa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f2389de34a4bfaca378ee750e68aa13
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_09d350bd367974f4fe007ab94f2e3aa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f2389de34a4bfaca378ee750e68aa13
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ec023f2cae487bea453c392c5faa5c7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a036e8ef9b445c8c5a36967b502be32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec023f2cae487bea453c392c5faa5c7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e95c3e0f03703f02d10e55336db462e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 4, 7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_81044dec829f0930a4d4ea3e7c54ba0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95c3e0f03703f02d10e55336db462e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6f5c7bc1f1a6e91ca57c8c4f2f903c72(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 20, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_51447e66f6520b898d28390a45886378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f5c7bc1f1a6e91ca57c8c4f2f903c72
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_95865149ae928955c8c1493f34fe090c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a79f499315df5df576d4fd603e601692(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95865149ae928955c8c1493f34fe090c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_3239bc132f4687efc08dd3a1752ff8ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 60800, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c77a57da35998b57bf1a42298d681c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3239bc132f4687efc08dd3a1752ff8ba
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4c77a57da35998b57bf1a42298d681c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3239bc132f4687efc08dd3a1752ff8ba
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_700b4e7cbddb96f49ddd5c34c70c3873(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a19bf0b09f59233f09c13081f7b9e9f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_700b4e7cbddb96f49ddd5c34c70c3873
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a19bf0b09f59233f09c13081f7b9e9f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_700b4e7cbddb96f49ddd5c34c70c3873
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bfc42a3cad9f0d1754fcce21d3e141fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11b890bb65900f90f2880f27a9556e64
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bfc42a3cad9f0d1754fcce21d3e141fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11b890bb65900f90f2880f27a9556e64
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_539eb28b72080f0324309899f6d38cbc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9702a96bb53675847a2bb1753d13b710(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539eb28b72080f0324309899f6d38cbc
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_84b6f74d4bf17c6e7d8b8d2b631190f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 784, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ed84379f25c24669dd19bb76023ec9fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84b6f74d4bf17c6e7d8b8d2b631190f2
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ed84379f25c24669dd19bb76023ec9fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84b6f74d4bf17c6e7d8b8d2b631190f2
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c66a8f6dc8bd9491918922599fe130f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21504, 1, 91], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1f6f8770888bf8e3b1f2fe9ac466f865(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c66a8f6dc8bd9491918922599fe130f2
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 1, 91], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_51c606d217445b0df9bf3c49fe414c8b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c5c04cbcfb8dc8133a9b9a45d780ba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51c606d217445b0df9bf3c49fe414c8b
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8c5c04cbcfb8dc8133a9b9a45d780ba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51c606d217445b0df9bf3c49fe414c8b
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_59dbb9a7466e0dbb691c48dbdc49e0b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 9216, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_338406d98c89f20235e97878b082be97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59dbb9a7466e0dbb691c48dbdc49e0b3
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_338406d98c89f20235e97878b082be97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59dbb9a7466e0dbb691c48dbdc49e0b3
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8532bc7d77461b35b45702f12f058256(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 21, 21], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_407464cff4685e953c55b96ca9d6c21a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8532bc7d77461b35b45702f12f058256
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 21, 21], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_478a88a54ca2bc5cf0bdbbf4824d9457(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 46, 46], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_112a951f76521f4ca430767263e9b310(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_478a88a54ca2bc5cf0bdbbf4824d9457
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 46, 46], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_01ab21c0a12f87c18055da17b6a48ff8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[551], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_67c719a9df77f5be568f5b1024197d1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01ab21c0a12f87c18055da17b6a48ff8
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_d24402dbdb89f4f65804ead474b69947(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0acf712efb0207bb7443f2f9edcb5940
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d24402dbdb89f4f65804ead474b69947(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0acf712efb0207bb7443f2f9edcb5940
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_386f3db1da08f46b5957227b5ac0d4b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d82becd1bcbd180ba17c48dca58ecd
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_386f3db1da08f46b5957227b5ac0d4b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d82becd1bcbd180ba17c48dca58ecd
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_fcbf8ea6c179a7dcd7e20cc101b5fe30(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 256, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74bdbf44a91110a0acd0bf6238e7fe35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fcbf8ea6c179a7dcd7e20cc101b5fe30
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 256, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bfc42a3cad9f0d1754fcce21d3e141fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11b890bb65900f90f2880f27a9556e64
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bfc42a3cad9f0d1754fcce21d3e141fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11b890bb65900f90f2880f27a9556e64
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d824401083df1fba2d8437cfdcc83c90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 80, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1f5719f963eb5acbddefdb13da9ed293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d824401083df1fba2d8437cfdcc83c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_386f3db1da08f46b5957227b5ac0d4b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d82becd1bcbd180ba17c48dca58ecd
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_386f3db1da08f46b5957227b5ac0d4b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d82becd1bcbd180ba17c48dca58ecd
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7ef3ae0669b9fb41cae8ffcefd402644(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 13, 13], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ae4fdd3ca69a5a2ff14e4bbf98c8c9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ef3ae0669b9fb41cae8ffcefd402644
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_ba7fac6aed21fd56332e03dc33467912(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 160, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bca91147d009742141a3155b2aae1c08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba7fac6aed21fd56332e03dc33467912
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bca91147d009742141a3155b2aae1c08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba7fac6aed21fd56332e03dc33467912
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_57a9a789310fcd0eee135db4efd634f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 160, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_218efbcd06d2e1e135397c0e3b2c39cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57a9a789310fcd0eee135db4efd634f2
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6b6125b5052c07d802233f97f91cac8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c8f6b76b4b5f82ec045b6a7685098a9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b6125b5052c07d802233f97f91cac8d
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_c555c2356a6131534428ad1461577b26(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 169, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e4f91c57182d54086461257b4cba719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c555c2356a6131534428ad1461577b26
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e4f91c57182d54086461257b4cba719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c555c2356a6131534428ad1461577b26
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e4f91c57182d54086461257b4cba719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c555c2356a6131534428ad1461577b26
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e4f91c57182d54086461257b4cba719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c555c2356a6131534428ad1461577b26
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_38252a922e4c6735563af4ed4ec4a1e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a6b6b607836f1d18b9d5d9d8e2e466de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38252a922e4c6735563af4ed4ec4a1e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d9944173c6970dc6ea3d7617f65f72b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2053, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3cb0238661339a92f8d64ee7006ba00f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9944173c6970dc6ea3d7617f65f72b9
    def get_inputs(self):
        return [
            paddle.uniform([2053, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_40fe0e4562a0c40b91d79fd98f62b022(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2053, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ca5029b4e7f857017b1a8063c72f58c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40fe0e4562a0c40b91d79fd98f62b022
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_56e4bd9ef05f7d147f6ff6e851e464bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2053, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_200a0f1d9c936c0be0768e725a337da0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56e4bd9ef05f7d147f6ff6e851e464bf
    def get_inputs(self):
        return [
            paddle.uniform([2053, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7ca5029b4e7f857017b1a8063c72f58c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40fe0e4562a0c40b91d79fd98f62b022
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_7bb21336ec456938c15b6d5b51f031da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 23, 23], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e89055bb774344fd90c8c19b31ad9cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bb21336ec456938c15b6d5b51f031da
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 23, 23], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_d8b36852dfb60e59fd7a3e2bec6f1ae4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f86e761e840153d6018e934970c383b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b36852dfb60e59fd7a3e2bec6f1ae4
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_4c717fa440ef63d40486c4fc886aa005(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[103, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1d70058a45ba3117626cdd76a016f2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c717fa440ef63d40486c4fc886aa005
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_1b207452b4e6e093b051171520ddbd04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32768, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_71c006a8f5980e873421648d86deb8f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b207452b4e6e093b051171520ddbd04
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_71c006a8f5980e873421648d86deb8f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b207452b4e6e093b051171520ddbd04
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1151fb0d19149b1cccb58c3bbf1f09b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 320, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b9687b5e9bb59f4b58ce466e2320946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1151fb0d19149b1cccb58c3bbf1f09b6
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7b9687b5e9bb59f4b58ce466e2320946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1151fb0d19149b1cccb58c3bbf1f09b6
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_08f20958bd5aa80eb7c9c85c295ac4ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 320, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dd3badb3e278ff9f2cdea91a6696688b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08f20958bd5aa80eb7c9c85c295ac4ce
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_bd9f86caa7bea6a9b04e70ef7f3767fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 96, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b9a96cadbcf2dd783b3caab0ee758762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd9f86caa7bea6a9b04e70ef7f3767fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 96, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_bda2ffddc8328c6339fde85cf3f726f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 16, 16, 49, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_15166e910e70c61e563cc078fb5ea280(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bda2ffddc8328c6339fde85cf3f726f8
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e83fb70fb7d4c0b6f21e4f97e1d8ad1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 112, 112], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_416d465b68448a02c55c599adfc837c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e83fb70fb7d4c0b6f21e4f97e1d8ad1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_815961350f8c7c1562fc0d786a813530(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e31f7a7cd49b575275f80bf145088a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_815961350f8c7c1562fc0d786a813530
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10706983506679535, -0.16371624171733856, -0.032012924551963806, -0.20100511610507965, 0.07308405637741089, -0.1075163334608078, 0.20525144040584564, -0.1600136160850525, -0.22037924826145172, -0.2549372911453247, 0.011028444394469261, -0.1464407742023468, -0.21996434032917023, 0.15681684017181396, -0.15918788313865662, -0.11360462754964828, 0.2256990224123001, 0.06460912525653839, -0.17191094160079956, -0.00266594928689301], dtype='float32').reshape([20]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_c3b7b5ce97ff5e94b53588910f1de370(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 65536, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e0ac1132a6e3ab13bc1a1aff471d6fb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3b7b5ce97ff5e94b53588910f1de370
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e0ac1132a6e3ab13bc1a1aff471d6fb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3b7b5ce97ff5e94b53588910f1de370
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c7926fbb1163fa47ec956abe564095b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_43e318b7b08c79d52bec829d2045193f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7926fbb1163fa47ec956abe564095b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_065f9abc4bc778c9c6357d3336b993cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1825, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_803952e0a40fe380f421fc44452321f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_065f9abc4bc778c9c6357d3336b993cb
    def get_inputs(self):
        return [
            paddle.uniform([1825, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_c607d3a8bdc7bbeddbd884a9e74efef6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1825, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c802084c4e6a06a422895b2fcf310108(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c607d3a8bdc7bbeddbd884a9e74efef6
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_22b64cbea84fe67da22b20b9c163848d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1825, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_19ba5addd236ae1e8a78305239f36c7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22b64cbea84fe67da22b20b9c163848d
    def get_inputs(self):
        return [
            paddle.uniform([1825, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c802084c4e6a06a422895b2fcf310108(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c607d3a8bdc7bbeddbd884a9e74efef6
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_e0978a011780d9e4da0bb0173786a951(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 34, 34], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9883f74f983ec54a3d7f449b62776344(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0978a011780d9e4da0bb0173786a951
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 34, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_40176fa4cd7b5ee67634a8116ee540a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 32, 16, 49, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8a7abe5ff7254405817614c90e3cd05e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40176fa4cd7b5ee67634a8116ee540a0
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a6c75704ef2586e01ac8d1da443b487a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1f937ce7a7ef7f09c26955f700e5b0e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6c75704ef2586e01ac8d1da443b487a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1f937ce7a7ef7f09c26955f700e5b0e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6c75704ef2586e01ac8d1da443b487a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4a7a631c9def817357cd95290cd2edee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b4d9c476c169cb3059330283143986ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a7a631c9def817357cd95290cd2edee
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0ac0f08df411b49b5e74c6c02dc9a9dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 19, 34], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_45d320322553a2928fd1c776ac797d89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ac0f08df411b49b5e74c6c02dc9a9dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_a27299759704567a560da451d04fc1bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a3b9fa8b770da7bc5fd4b908a6243b2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a27299759704567a560da451d04fc1bc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.19281575083732605, 0.03130542114377022, 0.22556477785110474, 0.08869007229804993, -0.00840307679027319, 0.25568798184394836, 0.15929333865642548, -0.06291423738002777, -0.16517488658428192, 0.18107402324676514, 0.17711132764816284, 0.11626065522432327, 0.12779568135738373, 0.0476863719522953, 0.09686526656150818, 0.11316271871328354], dtype='float32').reshape([16]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_d42d2c6d726d1d3853f2f7096cc13341(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4832], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bacad765a7e15b1790e6bb6177b257eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d42d2c6d726d1d3853f2f7096cc13341
    def get_inputs(self):
        return [
            paddle.uniform([4832], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_289976e111f86bf1c6ad1fb6092b895c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 2304, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_37939b46588fe45223baa1ba3d5092d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_289976e111f86bf1c6ad1fb6092b895c
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_37939b46588fe45223baa1ba3d5092d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_289976e111f86bf1c6ad1fb6092b895c
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8e6580cd6ee8b9f6d8299a541098cdbb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21760, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5aae7692b0f0ad1d5e22900847475559(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e6580cd6ee8b9f6d8299a541098cdbb
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5aae7692b0f0ad1d5e22900847475559(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e6580cd6ee8b9f6d8299a541098cdbb
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_02aafe989cc06cb2fa83f529a56fa316(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 16, 49, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3ce394037c7b20ad806293093ffb7701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02aafe989cc06cb2fa83f529a56fa316
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f5ffef4127f53d95dcd285b4684ef487(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 256, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ce9caed6fab38446ef30d1d29fc02691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5ffef4127f53d95dcd285b4684ef487
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 256, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_04ad5e1ffe49dde6ff94535f3e833f23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 16, 49, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ffc604e987060ba068042d0e0938ba8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04ad5e1ffe49dde6ff94535f3e833f23
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b70936cd8924db4d35730518447219f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 198, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_34620f205af07c117f70cd97b9dd956c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b70936cd8924db4d35730518447219f3
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_34620f205af07c117f70cd97b9dd956c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b70936cd8924db4d35730518447219f3
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_48e5dea8f3ba74d79c6d562c04f6bc31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 198, 198], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_562449085aa6f017e041a94c0694b8bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48e5dea8f3ba74d79c6d562c04f6bc31
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 198], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_687da036597c8cd539f152845eeafbc8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 19, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc0fab8af32d6b981c86850db1f4e494(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_687da036597c8cd539f152845eeafbc8
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_62774cf9c66ed8e892c0230584957966(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4594ec061a53037f77b63a3921c15d24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62774cf9c66ed8e892c0230584957966
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ae68aa33bee6c30e03b4b81d1ea446ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3087, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_167f831d8e4ed70efebac40a06bba6ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae68aa33bee6c30e03b4b81d1ea446ac
    def get_inputs(self):
        return [
            paddle.uniform([3087, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_331df2390754cb07f61c968e899ec6ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3087, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e0db9d47ca6e8b1bf77745ab342e997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_331df2390754cb07f61c968e899ec6ca
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_418272d580b409161f07c5d9fa6d0da0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3087, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f15fef530e96f5bd39a8facefcc926a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_418272d580b409161f07c5d9fa6d0da0
    def get_inputs(self):
        return [
            paddle.uniform([3087, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0e0db9d47ca6e8b1bf77745ab342e997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_331df2390754cb07f61c968e899ec6ca
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_5d86d494bf7925810ecdef0847fa6d32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240, 128, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e3615a35db537e16fa6df95008ac939f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d86d494bf7925810ecdef0847fa6d32
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_b10b33742f2f2148a4020fdd9090ee61(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_be7089c751462ba905a9109a5fddc137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b10b33742f2f2148a4020fdd9090ee61
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_50ff1fbdc208c39aaaf5ba2c26ab4d8b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_805f3a663ea42db81c8184f44ebec3cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50ff1fbdc208c39aaaf5ba2c26ab4d8b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_805f3a663ea42db81c8184f44ebec3cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50ff1fbdc208c39aaaf5ba2c26ab4d8b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e57679d89c2458b6a99d64793500f07a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe5cac74bbaa59835f4f7033082872b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ed84379f25c24669dd19bb76023ec9fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84b6f74d4bf17c6e7d8b8d2b631190f2
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ed84379f25c24669dd19bb76023ec9fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84b6f74d4bf17c6e7d8b8d2b631190f2
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_43d1192a566bc03f4a3dfd1b6457d858(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 128, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_24c1f5fe3604ac4b88805a43e2bb18f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43d1192a566bc03f4a3dfd1b6457d858
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_60e74f28a2c4219dac2d0509b124a47d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 52, 52], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f4acb914fec8cab4819775cdb2f2e2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60e74f28a2c4219dac2d0509b124a47d
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2e328f940e2e2c7b893f8ca9712a8d64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1044f941ffc5db497c999a20857202cf
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2e328f940e2e2c7b893f8ca9712a8d64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1044f941ffc5db497c999a20857202cf
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3ce394037c7b20ad806293093ffb7701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02aafe989cc06cb2fa83f529a56fa316
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1b21360c4d3c1b9bd5e3cccf5f881f3e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 68, 68], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e9adcb873595289597b3089004887b10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b21360c4d3c1b9bd5e3cccf5f881f3e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 68, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_78a45f5f17bbbaa50547c6a470768d08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b8a87c84c7f7ac866ff73ee3f9861cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_78a45f5f17bbbaa50547c6a470768d08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b8a87c84c7f7ac866ff73ee3f9861cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a71bad47fb68fee65842f23372609296(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 19, 32768], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f8c5846cf590aaed5be75c91bdd84b85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a71bad47fb68fee65842f23372609296
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 32768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_338406d98c89f20235e97878b082be97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59dbb9a7466e0dbb691c48dbdc49e0b3
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_338406d98c89f20235e97878b082be97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59dbb9a7466e0dbb691c48dbdc49e0b3
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_34cd5183222bc93e954d7879f3e9c085(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 60, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_853666390dc38983e5470945ba8ea04f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34cd5183222bc93e954d7879f3e9c085
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_4d6353cc366055b9253e9e177970cf42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e0ec8022023c4a3270bc0e875e0e8bf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d6353cc366055b9253e9e177970cf42
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e0ec8022023c4a3270bc0e875e0e8bf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d6353cc366055b9253e9e177970cf42
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f95ca53409b265af3d50ba9f34f5ba9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8f2ed709d5ede07eba99d8ceed79343
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_adc71242b7c01b531215ac4dc7a5ccb3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 50, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c67bd537a66b4d95279de501792f483(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adc71242b7c01b531215ac4dc7a5ccb3
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4c67bd537a66b4d95279de501792f483(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adc71242b7c01b531215ac4dc7a5ccb3
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_356ecbe9a447892d6d44e38441d0b0b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 80, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f7ce878974574cf5c95edc8034dadfe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_356ecbe9a447892d6d44e38441d0b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ab15839faa20888f2ea55a2ab6d6576c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dac9c99830c5b556862b0366080ff63d
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ab15839faa20888f2ea55a2ab6d6576c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dac9c99830c5b556862b0366080ff63d
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_31f79c91572dc9c6dcdd6ddd81ae954a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_962d18c8381e883806f74b905f660a93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31f79c91572dc9c6dcdd6ddd81ae954a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_78241108ce2f73caa51f008bc1824384(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2119, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_848c8eac180b316dc5e77c09643d7063(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78241108ce2f73caa51f008bc1824384
    def get_inputs(self):
        return [
            paddle.uniform([2119, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_4c829bae5d1af106a295caf8846756eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2119, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5beae60bf3bacf6cf54451bd6d11ea89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c829bae5d1af106a295caf8846756eb
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_f14faab026b3ff5adf90ab8074807c4d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2119, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9b44a11c073a4ffcaf35fb13b916d4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f14faab026b3ff5adf90ab8074807c4d
    def get_inputs(self):
        return [
            paddle.uniform([2119, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5beae60bf3bacf6cf54451bd6d11ea89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c829bae5d1af106a295caf8846756eb
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_8306124f9852faf04b0f448d4ed9b36a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 256, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_04deb4a129c70961fcf7b84ffdad8cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8306124f9852faf04b0f448d4ed9b36a
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 256, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_f242081abfb0e75e5b5fd0b98d18159a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 52, 52], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc9aba7dedcec1f700aeda9e47ca021c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f242081abfb0e75e5b5fd0b98d18159a
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8152c57d3fcf77df607db30fb0e32ef0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5092d496a9f82cc0016c5dd542d3883
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8152c57d3fcf77df607db30fb0e32ef0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5092d496a9f82cc0016c5dd542d3883
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4776dd711ebf7bc96542aa0220c21d4d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_29fc120578674262190e60a2646b71c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4776dd711ebf7bc96542aa0220c21d4d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_29fc120578674262190e60a2646b71c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4776dd711ebf7bc96542aa0220c21d4d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c080905de86684ab86f2bdb629349131(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1174, 1174], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7c8755ab4da5d233122d598b950daee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c080905de86684ab86f2bdb629349131
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_06a3d94bd9cb43f102d353b32bfcd458(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7dc06571f2bdff535e3265637f60de16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06a3d94bd9cb43f102d353b32bfcd458
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4ffc604e987060ba068042d0e0938ba8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04ad5e1ffe49dde6ff94535f3e833f23
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_47a0ac4d198e0b8c0fe99ccee52434c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[950], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92b7eed7095e03bfa2e2f8294d2d5e80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47a0ac4d198e0b8c0fe99ccee52434c5
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_2add538a2d929262f1721b1a3eb6615e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 288, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_548ad1350ab35c328d1af522dd030639(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2add538a2d929262f1721b1a3eb6615e
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_f52621ac2dcdf6bb29ec2f4019607614(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f688e406a3195034752a6e8aad688cab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52621ac2dcdf6bb29ec2f4019607614
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3020c4e00d84331d66536f0c6c20fda1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 198, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bee779e93d5ca14303e816b82f292fa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3020c4e00d84331d66536f0c6c20fda1
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bee779e93d5ca14303e816b82f292fa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3020c4e00d84331d66536f0c6c20fda1
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0e352d3992b743082fbd759dbbf19583(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 198, 198], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_029bfcf8b09a2962fba40a6528810035(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e352d3992b743082fbd759dbbf19583
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 198], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_aa2ff702f0d07115f21ff1f19cf85c20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_95d53162812fdd491760af09ec637c76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2ff702f0d07115f21ff1f19cf85c20
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_bab7ea657eabd2a88d22cb8961420480(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 576, 15, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2222e07354dad1498c9e82ae8ce0c2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bab7ea657eabd2a88d22cb8961420480
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 15, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_bb3267f35acc1b9b2f339389b808dd5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c06f176686b13803d3832ce19178cb7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb3267f35acc1b9b2f339389b808dd5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_386f3db1da08f46b5957227b5ac0d4b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d82becd1bcbd180ba17c48dca58ecd
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_386f3db1da08f46b5957227b5ac0d4b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d82becd1bcbd180ba17c48dca58ecd
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_368c13dc53af6c29bc0023a310da4a03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[247], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5ed978f67535eb5d244bc14c1d2e938e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_368c13dc53af6c29bc0023a310da4a03
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_39d18d3571e1925d3ca2c7f447e337b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 9216, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8df08424a8dd4f97d943994854ea72b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39d18d3571e1925d3ca2c7f447e337b9
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8df08424a8dd4f97d943994854ea72b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39d18d3571e1925d3ca2c7f447e337b9
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_cbe514d137dad4d4767c96df67c2584e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 288, 30, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fda1277b87d174bd8880d59b91041a16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbe514d137dad4d4767c96df67c2584e
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_97e365bf0a29cbd945910ce2b4d4e7fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c5a79677e0648b179391805a1acd124a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97e365bf0a29cbd945910ce2b4d4e7fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_b22ab0d6b0b86d34e2a04529d876c56e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc4be9ef073eca6a0260ed3a0a82c39c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b22ab0d6b0b86d34e2a04529d876c56e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_66fd0eba6c7aa2d343b1b7cd53741add(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 8192], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a736e8d1acf4bc46633e9bc4866635e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66fd0eba6c7aa2d343b1b7cd53741add
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 8192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a19bf0b09f59233f09c13081f7b9e9f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_700b4e7cbddb96f49ddd5c34c70c3873
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a19bf0b09f59233f09c13081f7b9e9f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_700b4e7cbddb96f49ddd5c34c70c3873
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8c5c04cbcfb8dc8133a9b9a45d780ba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51c606d217445b0df9bf3c49fe414c8b
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8c5c04cbcfb8dc8133a9b9a45d780ba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51c606d217445b0df9bf3c49fe414c8b
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4c67bd537a66b4d95279de501792f483(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adc71242b7c01b531215ac4dc7a5ccb3
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4c67bd537a66b4d95279de501792f483(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adc71242b7c01b531215ac4dc7a5ccb3
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_59379da71ac130783e0f6b7dd89ff234(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 50, 50], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa456d22ddfbc3b2262959988fb69c3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59379da71ac130783e0f6b7dd89ff234
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2595cc2ed6fda1a8397cce454a9e5727(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1dcfecd254d4e1e13bf591b056e5b63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2595cc2ed6fda1a8397cce454a9e5727
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_8152c57d3fcf77df607db30fb0e32ef0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5092d496a9f82cc0016c5dd542d3883
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8152c57d3fcf77df607db30fb0e32ef0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5092d496a9f82cc0016c5dd542d3883
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_17a54a89a77b5ba4b6c193d1a4200e1e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 160, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c2e6b597dfd3258cdfb4c718e9cd019b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17a54a89a77b5ba4b6c193d1a4200e1e
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_f5b68813265ae88868cf7570e2f32310(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 576, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_854014d311cdf725a9c028e44968a3d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5b68813265ae88868cf7570e2f32310
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_854014d311cdf725a9c028e44968a3d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5b68813265ae88868cf7570e2f32310
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3ff5fd2a105a7455a489c925f672da60(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1977326991aa591c900118c23be69ebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ff5fd2a105a7455a489c925f672da60
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1977326991aa591c900118c23be69ebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ff5fd2a105a7455a489c925f672da60
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_96eda4a16b4820236c9fd2d7a7f97880(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1025, 1025], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1ebaad7f0ff2638601d3aaa9cf80d0f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96eda4a16b4820236c9fd2d7a7f97880
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9c210a37f1e28628c208d601df98617b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 64, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_06c4c14fc7317109b61eea5221968d62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c210a37f1e28628c208d601df98617b
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_06c4c14fc7317109b61eea5221968d62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c210a37f1e28628c208d601df98617b
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_14178f7caf3b1e6e280ab86af1be5364(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 128, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8a60c828269678c344fb734fcaf6a4e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14178f7caf3b1e6e280ab86af1be5364
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8a60c828269678c344fb734fcaf6a4e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14178f7caf3b1e6e280ab86af1be5364
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_67e5d18e43af0cf05c1c4a28a2e485d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 256, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7841597482a0c991a067a61f5aefdba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e5d18e43af0cf05c1c4a28a2e485d5
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7841597482a0c991a067a61f5aefdba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e5d18e43af0cf05c1c4a28a2e485d5
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c3605226bca835c1abab7aeea1b5c8da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 512, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b5b7d4fc599f40ac55db9f6d8d7451d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3605226bca835c1abab7aeea1b5c8da
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b5b7d4fc599f40ac55db9f6d8d7451d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3605226bca835c1abab7aeea1b5c8da
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1eb263398df6cb90934d882fd022ad57(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 288, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_51eb166ef7fde75caf7da3bccc630eed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1eb263398df6cb90934d882fd022ad57
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_bfc695a5993629afe68939678f777cb3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240, 84, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_56247b3c8f84247d05e5c1b9a303f390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfc695a5993629afe68939678f777cb3
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 84, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_f44421e6662cea73d5b5bb9c640c9eb2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_78765aa17455851dac89cca848820d37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f44421e6662cea73d5b5bb9c640c9eb2
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_7efea92cd30f4704d50f47febc078666(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a2b35c13766a882e3c79845e4e333732(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7efea92cd30f4704d50f47febc078666
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_80be9cf6ac97b50650b4ecac2f261d33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3fd8b02a56827bcc0c7e95ed909dd8ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80be9cf6ac97b50650b4ecac2f261d33
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_03dbc6800f572e53591fef5726517b31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5606, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b40bb0a46e98497a5490701a31c00a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03dbc6800f572e53591fef5726517b31
    def get_inputs(self):
        return [
            paddle.uniform([5606, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_95ee31f27e3bf0620fdda8ace7e28c8b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5606, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_87c2e20de95610e81cb089ee3a77fd85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95ee31f27e3bf0620fdda8ace7e28c8b
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_dc0ba97626740e33be646ba16ee1205f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5606, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_387418f0b29e73ef08b01ccd2c06e1ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc0ba97626740e33be646ba16ee1205f
    def get_inputs(self):
        return [
            paddle.uniform([5606, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_87c2e20de95610e81cb089ee3a77fd85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95ee31f27e3bf0620fdda8ace7e28c8b
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_fe08e650af3e0d9d132cb3c25674a345(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1d2fb05f15f31a19b6e00e422107ac0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe08e650af3e0d9d132cb3c25674a345
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_054b4829561c2e4a651533683e9480f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 42, 42], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80d8239f68709765d888bd088df6a026(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_054b4829561c2e4a651533683e9480f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 42, 42], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_82fd2066a910ad1ab14e0108600f0e3d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc3defd7e3c25cd0c350d27780ef9854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82fd2066a910ad1ab14e0108600f0e3d
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_153a5d2a5f4c9a909164b52fedf95e1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_077eb1c6985ea591a8fa4bc551b68d24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_153a5d2a5f4c9a909164b52fedf95e1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_76c6c056d28aae3871c9b589035c2831(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1036, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2bfee88b952a535bd7e851ee0a1f72a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76c6c056d28aae3871c9b589035c2831
    def get_inputs(self):
        return [
            paddle.uniform([1036, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_25d9c77be6a422c6f7235de757ef60a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1036, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f0e48e2774988d71b1a7dbe4adfa6f3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25d9c77be6a422c6f7235de757ef60a8
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_5721e8c19eefc5aaf91fadc4fed50d9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1036, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c68972ca17fa40125378563c693bed44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5721e8c19eefc5aaf91fadc4fed50d9b
    def get_inputs(self):
        return [
            paddle.uniform([1036, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f0e48e2774988d71b1a7dbe4adfa6f3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25d9c77be6a422c6f7235de757ef60a8
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_00f34fe5927f6f79f4a2a301ffe19769(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[53, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1380519f6b9e045a07b451acd0d187e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00f34fe5927f6f79f4a2a301ffe19769
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e0ec8022023c4a3270bc0e875e0e8bf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d6353cc366055b9253e9e177970cf42
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e0ec8022023c4a3270bc0e875e0e8bf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d6353cc366055b9253e9e177970cf42
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_43e318b7b08c79d52bec829d2045193f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7926fbb1163fa47ec956abe564095b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_363fd9f6bded77ed588c55fb9ef9ec0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1809, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53437a833761cd8fcf64f873d97f959e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_363fd9f6bded77ed588c55fb9ef9ec0b
    def get_inputs(self):
        return [
            paddle.uniform([1809, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_9dd6a64e12b894010d5ebe9142cc4ab3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1809, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ddd49cf4061c8fc239f4620a312cec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dd6a64e12b894010d5ebe9142cc4ab3
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_9a505afb61fd9fcd2a2bb7e5219f722a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1809, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8a4a93a2edbfaafd6e74cb3b42885f29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a505afb61fd9fcd2a2bb7e5219f722a
    def get_inputs(self):
        return [
            paddle.uniform([1809, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ddd49cf4061c8fc239f4620a312cec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dd6a64e12b894010d5ebe9142cc4ab3
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_5f9d4ae1788e81eba556535798608b52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 197, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5dd6180682433de415c966766ede637e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f9d4ae1788e81eba556535798608b52
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5dd6180682433de415c966766ede637e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f9d4ae1788e81eba556535798608b52
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3671f359b0fb385bda6fafd90c243630(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 197, 197], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a44ccdacb78559e0b2a0186032ba51a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3671f359b0fb385bda6fafd90c243630
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 197], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2f86e761e840153d6018e934970c383b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b36852dfb60e59fd7a3e2bec6f1ae4
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_f754b4b075ae6f9655c86df7aac156b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[84, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97b0d19b224624d5051c093e614e974a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f754b4b075ae6f9655c86df7aac156b9
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_73f9eb52575ad454bde0c13047640015(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3ba08b9553d241b4727b5c69288a67c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73f9eb52575ad454bde0c13047640015
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_cfc73a367871f49d801fc81f00c1b010(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 4, 13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c7dd8ee82bd5dd79dda76357b9bd2f6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfc73a367871f49d801fc81f00c1b010
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0e62bc04ea33082de6b31f8834c03edf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 17, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c8549ef4b421b987c8bf10d2b9666d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e62bc04ea33082de6b31f8834c03edf
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 17, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_05796bdd8729c9d48b2c3ef6c7b4e0a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 13, 13], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6199f6076caceb331e83555cdf352bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05796bdd8729c9d48b2c3ef6c7b4e0a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_d9f1887748efc34607438842a50f1e5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf52711d7a4df0586f9f9e07b7d59713(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9f1887748efc34607438842a50f1e5c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11860727518796921, 0.003984234295785427, 0.17628856003284454, -0.051732927560806274, -0.010178853757679462, 0.13205716013908386, -0.07914058119058609, -0.043340764939785004, -0.188715860247612, 0.27194005250930786, -0.3161042034626007, -0.11421764642000198, -0.19390501081943512, 0.28144291043281555, -0.16818910837173462, -0.05955290049314499, 0.2610541582107544, 0.2058759480714798, -0.20685532689094543, -0.08991886675357819, -0.03967750445008278, -0.09968677908182144, 0.23027987778186798, 0.04034080356359482], dtype='float32').reshape([24]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_8a7abe5ff7254405817614c90e3cd05e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40176fa4cd7b5ee67634a8116ee540a0
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5f053a0f329371bd150d1c536507da92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 4096], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a4c2bc6cb14e4c03ff3e40bb40a01842(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f053a0f329371bd150d1c536507da92
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 4096], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_513a6c57dbe4e8c086a9163b359acb29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17421], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ccdd2ecb7797376e823dcc86bd561bd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_513a6c57dbe4e8c086a9163b359acb29
    def get_inputs(self):
        return [
            paddle.uniform([17421], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_20abee4dad939004c06936a96334fcf3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 16, 9, 112, 112], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c587d4ca6f3b238d285bf647c39c0632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20abee4dad939004c06936a96334fcf3
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_029d505d74c71e5a85962f6ca67758d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ec9cd30550a4fbd82462c3858b2cbc7
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_029d505d74c71e5a85962f6ca67758d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ec9cd30550a4fbd82462c3858b2cbc7
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_61b86c5aa3ce912a964d107c206ac35a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad5b49f096c37ec3bca9ec38824a3b77
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_61b86c5aa3ce912a964d107c206ac35a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad5b49f096c37ec3bca9ec38824a3b77
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_da7163a9e58d18300c1d10293015de31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29a3ed6ebd61206e3d2cd6f6c680b8dc
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_da7163a9e58d18300c1d10293015de31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29a3ed6ebd61206e3d2cd6f6c680b8dc
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3d63f530a5d49615277ac5ab96283351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b7fdaa99413d75d967428fc12f38563
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3d63f530a5d49615277ac5ab96283351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b7fdaa99413d75d967428fc12f38563
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_194a794c2be933fb2ba9ea93d2c93341(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 577, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bac4359e66193ea1e204caac331436f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_194a794c2be933fb2ba9ea93d2c93341
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bac4359e66193ea1e204caac331436f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_194a794c2be933fb2ba9ea93d2c93341
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_238ca3068a19d01a8abf59c2cedacca0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 577, 577], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_465ccd3df2cd76d17a8e3276f37a08c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_238ca3068a19d01a8abf59c2cedacca0
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a74c7347b11a2030401682c8763dbff5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00d7d9283dd4c29fb1dd1d6d8613fda8
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d3cac74578dc0dac8be74d0a72896345(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2204], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f5b40b84f6ece16796107b9fdcbd3377(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3cac74578dc0dac8be74d0a72896345
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_8152c57d3fcf77df607db30fb0e32ef0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5092d496a9f82cc0016c5dd542d3883
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8152c57d3fcf77df607db30fb0e32ef0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5092d496a9f82cc0016c5dd542d3883
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9702a96bb53675847a2bb1753d13b710(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539eb28b72080f0324309899f6d38cbc
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c514c7c32e135bfb87674036839b0a6b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e2ed65e85216d8575dfb580bc24875a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c514c7c32e135bfb87674036839b0a6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_9814e7ee9e2648e094645b271b4a8d0c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f42436fae0ba62fb57ad695f3b9b157(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9814e7ee9e2648e094645b271b4a8d0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c7c2908c1da721f2d7626d63207aa91b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 4, 25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_85bae03d46122febe9ec8e8ab77b3ab0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c2908c1da721f2d7626d63207aa91b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d24402dbdb89f4f65804ead474b69947(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0acf712efb0207bb7443f2f9edcb5940
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d24402dbdb89f4f65804ead474b69947(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0acf712efb0207bb7443f2f9edcb5940
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8df08424a8dd4f97d943994854ea72b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39d18d3571e1925d3ca2c7f447e337b9
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8df08424a8dd4f97d943994854ea72b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39d18d3571e1925d3ca2c7f447e337b9
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_06c4c14fc7317109b61eea5221968d62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c210a37f1e28628c208d601df98617b
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_06c4c14fc7317109b61eea5221968d62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c210a37f1e28628c208d601df98617b
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8a60c828269678c344fb734fcaf6a4e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14178f7caf3b1e6e280ab86af1be5364
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8a60c828269678c344fb734fcaf6a4e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14178f7caf3b1e6e280ab86af1be5364
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7841597482a0c991a067a61f5aefdba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e5d18e43af0cf05c1c4a28a2e485d5
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7841597482a0c991a067a61f5aefdba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e5d18e43af0cf05c1c4a28a2e485d5
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b5b7d4fc599f40ac55db9f6d8d7451d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3605226bca835c1abab7aeea1b5c8da
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b5b7d4fc599f40ac55db9f6d8d7451d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3605226bca835c1abab7aeea1b5c8da
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ab15839faa20888f2ea55a2ab6d6576c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dac9c99830c5b556862b0366080ff63d
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ab15839faa20888f2ea55a2ab6d6576c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dac9c99830c5b556862b0366080ff63d
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bca91147d009742141a3155b2aae1c08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba7fac6aed21fd56332e03dc33467912
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bca91147d009742141a3155b2aae1c08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba7fac6aed21fd56332e03dc33467912
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9305daf88474abe13347c70469ab39cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 104, 104], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4faf5b74287242f2915761e3fa10d04d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9305daf88474abe13347c70469ab39cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 104, 104], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_4f9018b41f973066df108823e995a6b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 152, 272], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_99ed2bec3c2024eab46127bf9954cb5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f9018b41f973066df108823e995a6b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_ce79c2215932d7c10cfb9acd4d030e70(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 104, 104], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9406e92169c5d37b077245786edc8415(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce79c2215932d7c10cfb9acd4d030e70
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 104, 104], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8c5c04cbcfb8dc8133a9b9a45d780ba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51c606d217445b0df9bf3c49fe414c8b
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8c5c04cbcfb8dc8133a9b9a45d780ba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51c606d217445b0df9bf3c49fe414c8b
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3ec1c94556562e203461a918faca3bd8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 144, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7899b9ce5d962dea7ca6d559c6cc380d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ec1c94556562e203461a918faca3bd8
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7899b9ce5d962dea7ca6d559c6cc380d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ec1c94556562e203461a918faca3bd8
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_806e1dade052ac0f1a7f5dbd9d9d9f24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 184, 184], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b6cbfedd9f580db3e7821322cffd0ab1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_806e1dade052ac0f1a7f5dbd9d9d9f24
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 184], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_061e2f6c77ec0d3beb825a56582b854d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_323d19446faca91132f2906871d32852
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_0265288a369f00b23f64ce0cb9c3cd83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[150], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_65244a23fc02f7983171453648573ec6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0265288a369f00b23f64ce0cb9c3cd83
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_7b9687b5e9bb59f4b58ce466e2320946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1151fb0d19149b1cccb58c3bbf1f09b6
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7b9687b5e9bb59f4b58ce466e2320946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1151fb0d19149b1cccb58c3bbf1f09b6
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_09d350bd367974f4fe007ab94f2e3aa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f2389de34a4bfaca378ee750e68aa13
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_09d350bd367974f4fe007ab94f2e3aa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f2389de34a4bfaca378ee750e68aa13
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15166e910e70c61e563cc078fb5ea280(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bda2ffddc8328c6339fde85cf3f726f8
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_fa6b25b03db7cde1f2d7346f1b10e254(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 76, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_41c5c3adae31ca710478bbc4603db772(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa6b25b03db7cde1f2d7346f1b10e254
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 76, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8c5c04cbcfb8dc8133a9b9a45d780ba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51c606d217445b0df9bf3c49fe414c8b
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8c5c04cbcfb8dc8133a9b9a45d780ba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51c606d217445b0df9bf3c49fe414c8b
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c8e2299a28c41b31e06c101878243986(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e626dcaf1b2fe97297cde4de5be698a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e2299a28c41b31e06c101878243986
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3a54a84d6a65397c41b9d467de8b946a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4179, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3c532d928c7ffc57d9280797b8a6b79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a54a84d6a65397c41b9d467de8b946a
    def get_inputs(self):
        return [
            paddle.uniform([4179, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_0892ceab85b4198d40089c5162af0755(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4179, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0ba74e0b6a1ee26d22d137e04a13d123(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0892ceab85b4198d40089c5162af0755
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_848be8afcb5382e0e2795966e2a3fc4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4179, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a4707169f35d0efb66446ce93f278c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_848be8afcb5382e0e2795966e2a3fc4c
    def get_inputs(self):
        return [
            paddle.uniform([4179, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0ba74e0b6a1ee26d22d137e04a13d123(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0892ceab85b4198d40089c5162af0755
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_ec106e43997a91fd2f1d43134a12d60f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 256, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e907a1475b9b1ccd445c827bbf00d0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec106e43997a91fd2f1d43134a12d60f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 256, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bfc42a3cad9f0d1754fcce21d3e141fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11b890bb65900f90f2880f27a9556e64
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bfc42a3cad9f0d1754fcce21d3e141fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11b890bb65900f90f2880f27a9556e64
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0b8cbf698f2c7dac900cc52f8c96236f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ea3611aeb366f7840b37c7519e648552(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b8cbf698f2c7dac900cc52f8c96236f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d24402dbdb89f4f65804ead474b69947(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0acf712efb0207bb7443f2f9edcb5940
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d24402dbdb89f4f65804ead474b69947(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0acf712efb0207bb7443f2f9edcb5940
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e8af63b8085dbd5c04c5bc180b410ca9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_654893fc7ad4cf45eec24804d38d4033(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8af63b8085dbd5c04c5bc180b410ca9
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_8a461dfc55d1a85428f2fcf6addf8ecd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[52, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_72e742e86d00e677428043a3a3368282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a461dfc55d1a85428f2fcf6addf8ecd
    def get_inputs(self):
        return [
            paddle.uniform([52, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_b1839714bf19a77301dba4f96007ed07(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 128, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b577dfec543ca71b061cfbe6fc043b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1839714bf19a77301dba4f96007ed07
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_ef5239388d10e67b4e198de67e3fea8f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02b9f26e9c94812e14a7e7494192b49a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef5239388d10e67b4e198de67e3fea8f
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_02b9f26e9c94812e14a7e7494192b49a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef5239388d10e67b4e198de67e3fea8f
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_95671fa07f0e263f5d851e0ad70c8fb3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 92, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e24cb6f1314505dab895226b2210ab27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95671fa07f0e263f5d851e0ad70c8fb3
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 92, 92], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7b9687b5e9bb59f4b58ce466e2320946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1151fb0d19149b1cccb58c3bbf1f09b6
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7b9687b5e9bb59f4b58ce466e2320946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1151fb0d19149b1cccb58c3bbf1f09b6
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd3badb3e278ff9f2cdea91a6696688b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08f20958bd5aa80eb7c9c85c295ac4ce
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_838dc9eb8647ca6069c59ca961fa4df6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 144, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bae94effcb5c57bda2bdeec0f09e6a7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_838dc9eb8647ca6069c59ca961fa4df6
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bae94effcb5c57bda2bdeec0f09e6a7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_838dc9eb8647ca6069c59ca961fa4df6
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a27196e4b9132d5357adf34050e1c6fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 20, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2d6e12f563e93a071008216da2cc0c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a27196e4b9132d5357adf34050e1c6fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_46d07dcb40220ee154e558100391e528(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3800], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ebd9cbc492b83dc41f9eca9157fe7034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46d07dcb40220ee154e558100391e528
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_8a292a89346751d09e65408be02e49ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ce2da88fc0d088a9cdfb02f82c3cd630(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a292a89346751d09e65408be02e49ba
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_035dda936b7a6a29a6a06244af66b066(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4662, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d776850fddb2f537e7a22dff4515b84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_035dda936b7a6a29a6a06244af66b066
    def get_inputs(self):
        return [
            paddle.uniform([4662, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_26c478d58bc8e9c2144034bda0499ca6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4662, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f7ab8d7c6ae3fac0d865abc1d5f9e839(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c478d58bc8e9c2144034bda0499ca6
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_0abcc4ce583e262c5cafd003fa56f5f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4662, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ea87df73a21bf9a2647df2b23d5edad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0abcc4ce583e262c5cafd003fa56f5f8
    def get_inputs(self):
        return [
            paddle.uniform([4662, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f7ab8d7c6ae3fac0d865abc1d5f9e839(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c478d58bc8e9c2144034bda0499ca6
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_a0d24709b19bddd98990bfc087cff227(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539eb28b72080f0324309899f6d38cbc
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a0d24709b19bddd98990bfc087cff227(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539eb28b72080f0324309899f6d38cbc
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b4d9c476c169cb3059330283143986ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a7a631c9def817357cd95290cd2edee
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ed84379f25c24669dd19bb76023ec9fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84b6f74d4bf17c6e7d8b8d2b631190f2
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ed84379f25c24669dd19bb76023ec9fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84b6f74d4bf17c6e7d8b8d2b631190f2
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_02b9f26e9c94812e14a7e7494192b49a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef5239388d10e67b4e198de67e3fea8f
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_02b9f26e9c94812e14a7e7494192b49a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef5239388d10e67b4e198de67e3fea8f
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7aeedbfea6f22ce056568b76a1c0e99d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1ce059f86197f878a2564128c2c7d60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7aeedbfea6f22ce056568b76a1c0e99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_17993e69d2343e2cf0c05a7d95491d2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3857, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f0a79effda24b322508951ac8988ab55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17993e69d2343e2cf0c05a7d95491d2c
    def get_inputs(self):
        return [
            paddle.uniform([3857, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_68f122b3b3f7c2e4e100bcd30f706d7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3857, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5ed15cdf31f115875e4b36cb8ffe87da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68f122b3b3f7c2e4e100bcd30f706d7c
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_8ae9aed645ffadd04e373d8bd789356a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3857, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_98d9bfc6ce75cb8afbe1ac22f6636a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ae9aed645ffadd04e373d8bd789356a
    def get_inputs(self):
        return [
            paddle.uniform([3857, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5ed15cdf31f115875e4b36cb8ffe87da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68f122b3b3f7c2e4e100bcd30f706d7c
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_78a45f5f17bbbaa50547c6a470768d08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b8a87c84c7f7ac866ff73ee3f9861cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_78a45f5f17bbbaa50547c6a470768d08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b8a87c84c7f7ac866ff73ee3f9861cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4cc85c48c943528f22137ac727281009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2365f9505ee300638e59dcb407eb5d8
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f73a25313976c4ac0308ebad61a6e087(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_76517f4ff6206b4aaceca8823093c586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f73a25313976c4ac0308ebad61a6e087
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_76517f4ff6206b4aaceca8823093c586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f73a25313976c4ac0308ebad61a6e087
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d7b58de105ef85f13b053216b074287c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1174, 1174], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_514dd8384aa080999eecbca60119326f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7b58de105ef85f13b053216b074287c
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_336f203d01d11d8a27c32caef241a04d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b42bb266171953003a314000a32c69b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_336f203d01d11d8a27c32caef241a04d
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_45e621f75821261e5397acc69c92dd79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 48, 48], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3afc83e977eab8a51fccca87d8c7ec99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45e621f75821261e5397acc69c92dd79
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_163b7bcbeae454b0d633cf36b2a9fc2f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 4, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e5f26f890bc665d0ce3527ce700b3d24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_163b7bcbeae454b0d633cf36b2a9fc2f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6b7c5d16b9c29326492e7037591ca9a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 38, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_77b187a82e5a52cba0f3553f10c6e281(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b7c5d16b9c29326492e7037591ca9a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_f9cf0e78208ea7a65c4e56c5cb444d24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b52f9b42fffa1c644b06f957863595b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9cf0e78208ea7a65c4e56c5cb444d24
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c92256012b7ca583c1dd3d672cc5db7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89d93a4298ffafba0d72da3f76a95cc3
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c92256012b7ca583c1dd3d672cc5db7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89d93a4298ffafba0d72da3f76a95cc3
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0a686095ae548b432788207b215d59c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 10, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_44c924e70826cce523b04fd5b0b58054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a686095ae548b432788207b215d59c2
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_b98750bf4ebeb73fe0dcdf45aa7a6c30(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 152, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e5083198438c4f0c9d163ae49f9c33e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b98750bf4ebeb73fe0dcdf45aa7a6c30
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 152, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_d9bf5233eade29e28af5195132119b4e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 576, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a3acbd1a45153493913511343ad09faa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9bf5233eade29e28af5195132119b4e
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a3acbd1a45153493913511343ad09faa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9bf5233eade29e28af5195132119b4e
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_bed3b9466738fd650ab7eaccb317fa36(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_454e2197b6a4d3f43d04447d04187454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed3b9466738fd650ab7eaccb317fa36
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_12576afce880d9ada2bb60c4acaf1bf6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 16, 49, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d11581247341eb8e41a215b84dcaee25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12576afce880d9ada2bb60c4acaf1bf6
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_85389c017f5b0c35e6bbc9709834a3ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 128, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d5890c8a33815bd7708ea23e3d76dd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85389c017f5b0c35e6bbc9709834a3ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_4c47400cfd643ef4718c667219c56e10(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_29ea1b61b51287c9e54487c838cb6aa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c47400cfd643ef4718c667219c56e10
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_2cabcd9026e83a963a20fea7500431b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 38, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_368b4e6ccb38182ca107b919c229f564(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cabcd9026e83a963a20fea7500431b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c92256012b7ca583c1dd3d672cc5db7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89d93a4298ffafba0d72da3f76a95cc3
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c92256012b7ca583c1dd3d672cc5db7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89d93a4298ffafba0d72da3f76a95cc3
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()