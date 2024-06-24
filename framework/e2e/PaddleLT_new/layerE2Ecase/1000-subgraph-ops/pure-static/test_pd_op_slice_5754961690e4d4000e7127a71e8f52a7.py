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



class PrimitiveOp_6f1169f9071ce5743e24091fa800b817(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3d0f29d1519d89c77da331117fbd5a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 24, 36], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f4d823e1fd9c11cf015f59b1a3e0a9a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 24, 36], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d8f87f669abcdb86c1b01768647cb8fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 11, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_be6f74564f94616fd9b3f8822846cefb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8f87f669abcdb86c1b01768647cb8fe
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_554beb005b034e8d261f1b8cb829dfce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8f87f669abcdb86c1b01768647cb8fe
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b1f412f03d27b0c1f386d736846ac8d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 300, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_579c7c9c1a8d0112d0b63ac9388217a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1f412f03d27b0c1f386d736846ac8d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_123f4206f30e0c7376b3076b177cb920(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 11, 64, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2dfab79e7c5d696f69db561f6f97d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123f4206f30e0c7376b3076b177cb920
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_593713920e5f789036f4ed4537953915(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123f4206f30e0c7376b3076b177cb920
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_db12eccaf083484153a4cc1dd1486aa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123f4206f30e0c7376b3076b177cb920
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b625734245499b5ce7e9f95a6ca197fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 10, 4, 100, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b3ad386b33b73b61f841b1c98e9c30f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b625734245499b5ce7e9f95a6ca197fa
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ef9101d31f213bd62f7186855d0134dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b625734245499b5ce7e9f95a6ca197fa
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e67c9d49e6651db4cadd688757b70835(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b625734245499b5ce7e9f95a6ca197fa
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4f104df590b121e14c46663c1526d4d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 54, 3, 198, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_58fd0c2971d796cd80a3a1c7b4fa9173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f104df590b121e14c46663c1526d4d6
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_822f1db81002081e4d8137026db25ed0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f104df590b121e14c46663c1526d4d6
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8106130ebbcbca5e462ee1e7f0f30bec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f104df590b121e14c46663c1526d4d6
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0aacfe9c3b4c283dd9c968065a11802c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1960, 4, 16, 6], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b914f79436ba2e901437cbdea557204(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aacfe9c3b4c283dd9c968065a11802c
    def get_inputs(self):
        return [
            paddle.uniform([2, 1960, 4, 16, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3dbaae6d1bdda1a6a0fd1c90601bd83a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aacfe9c3b4c283dd9c968065a11802c
    def get_inputs(self):
        return [
            paddle.uniform([2, 1960, 4, 16, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7200a021039ba11b103f6e15d77dbd2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0fa1c5f7e126a65a13f3fe48a72215a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2df5271672b80afec4406dc57fe7a1af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0b4446cc8e877b119dc687e9e982910f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_cba436843473132c18085e1a14d5ef35(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 43, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3c3f3473be1a4ae1c87ddaa779f06640(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cba436843473132c18085e1a14d5ef35
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_17b6d771a7c41c1b551198fee689e039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cba436843473132c18085e1a14d5ef35
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3d0f13c8e1c2beac5ede1a7e45f8d6d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_24381d865a39514c999d25ecc4bf541d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ce0bdaed88c94597018697e65bb0217c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d2baeeb6c9f3b18f147d9cd9bc0432a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1250c64f7f0544b0426eb0bb6d0b2c23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e24d3112935e5a03677411939abf5f44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_00817a64ead21517203630173d3cb460(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4f063157bd604da0ae2ae05c617410ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 43, 64, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ee9f884eca2c6e32fcb5c82ebaf6ffd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f063157bd604da0ae2ae05c617410ad
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a8a44decdee11e8266da42f588050149(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f063157bd604da0ae2ae05c617410ad
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cc5069a45751f63b344845c741cb95ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f063157bd604da0ae2ae05c617410ad
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_610cb639151f5a8b06308abe403db8a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 76], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d589391d4f0aa09f1212116ceb3c4ef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 76], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_da01bd16d1cae63b273af0c17d07badf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 76], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_25a32f364be766f1d5431192fac4f68b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 576, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8fba6b55cebb7f81033828a739d45fb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 576, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_963af40d41f9d7c99a1316a80208c482(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 576, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_71c646d0d63ed8ee9a8642366d923ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 576, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0337b8de5f6b0b37b240449a32d19fa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 288, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f55ff2fb55ad2507e15651911c3f597c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 288, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9f19fc8f36513789d84266d7cc412fa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 288, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5a63f87907975ae2300fc3df10c4396b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 288, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1a3088d6f9b42bbc91b8cab7bc42b640(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_37730242d97d3d3a9314abee92e956d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_deade7c930455b4e37195490ac2762ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f6e73887fd748b5df0ef1a3949b1a773(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a25656aa6082221b78367e1f8359591a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_346f033e08de1e24cdbe389d461243ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a25656aa6082221b78367e1f8359591a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.38087019324302673, 0.0610022246837616, 0.02883603796362877, 0.12455703318119049]], [[0.1897112876176834, 0.1157618910074234, 0.44941291213035583, 0.31380409002304077]], [[0.20878849923610687, 0.19586940109729767, 0.2543567717075348, 0.06301438808441162]], [[0.33474111557006836, 0.41102105379104614, 0.14262086153030396, 0.009150227531790733]], [[0.27764520049095154, 0.3343753516674042, 0.24400947988033295, 0.3703606128692627]], [[0.23894988000392914, 0.048072364181280136, 0.3630118668079376, 0.31833693385124207]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_24710aa07c641e4812486a708fdb1af6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a25656aa6082221b78367e1f8359591a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.38087019324302673, 0.0610022246837616, 0.02883603796362877, 0.12455703318119049]], [[0.1897112876176834, 0.1157618910074234, 0.44941291213035583, 0.31380409002304077]], [[0.20878849923610687, 0.19586940109729767, 0.2543567717075348, 0.06301438808441162]], [[0.33474111557006836, 0.41102105379104614, 0.14262086153030396, 0.009150227531790733]], [[0.27764520049095154, 0.3343753516674042, 0.24400947988033295, 0.3703606128692627]], [[0.23894988000392914, 0.048072364181280136, 0.3630118668079376, 0.31833693385124207]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_979d625450cc8bbfed7fbdd178cd47ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a25656aa6082221b78367e1f8359591a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.38087019324302673, 0.0610022246837616, 0.02883603796362877, 0.12455703318119049]], [[0.1897112876176834, 0.1157618910074234, 0.44941291213035583, 0.31380409002304077]], [[0.20878849923610687, 0.19586940109729767, 0.2543567717075348, 0.06301438808441162]], [[0.33474111557006836, 0.41102105379104614, 0.14262086153030396, 0.009150227531790733]], [[0.27764520049095154, 0.3343753516674042, 0.24400947988033295, 0.3703606128692627]], [[0.23894988000392914, 0.048072364181280136, 0.3630118668079376, 0.31833693385124207]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c9ef82f4f4ce5f71733192e3785d81fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a25656aa6082221b78367e1f8359591a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.38087019324302673, 0.0610022246837616, 0.02883603796362877, 0.12455703318119049]], [[0.1897112876176834, 0.1157618910074234, 0.44941291213035583, 0.31380409002304077]], [[0.20878849923610687, 0.19586940109729767, 0.2543567717075348, 0.06301438808441162]], [[0.33474111557006836, 0.41102105379104614, 0.14262086153030396, 0.009150227531790733]], [[0.27764520049095154, 0.3343753516674042, 0.24400947988033295, 0.3703606128692627]], [[0.23894988000392914, 0.048072364181280136, 0.3630118668079376, 0.31833693385124207]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8fb1a0239b0d9299a9e0cc4a8c2e7cff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 100, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0415765b2174cc5293ec150cc550160a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fb1a0239b0d9299a9e0cc4a8c2e7cff
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2e74110f99ed1185ec21c5ffa288e990(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_27e9dc5646722f4a7d4e7b42c4e56ff9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4b150b07f8df9e3c381bae373a0bb54c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6992aaabc2c028932535401b2fc8c563(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6351370c107ca3bf22f9d660f220a57e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ad23d257a30c5cc7ee3f9aaf8fb81ef2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cd2f85d246eb5e988d48a9b1894a4e7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_36a44245746314e0c457e0ea4a8d578c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b228f16c45a28cdf8d2ec691cda20303(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_011a01ada928324acec7bf36f3c0801a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_829887b03bde77c51b681ba316d3074c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 43, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df3c151fec3377785fa1e1a6d2ff2ade(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_829887b03bde77c51b681ba316d3074c
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_65a25d64388215630a00a9932e70d10c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_829887b03bde77c51b681ba316d3074c
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_095c3d132ce0dc468df81e6e3d854c7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([128, 320, 8, 4], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_01fefbcbad20a0c2ef365d3c66e6d61d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([128, 320, 8, 4], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_875e62b941d24344a9aaa17e35dc1654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([128, 320, 8, 4], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c5ed2d77a2b0ce54ac26ddbc332e6996(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2641b71fe698c28e497ab88cc823d4d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d0133439bdec1d8ca77d0e9764f05d35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 7, 7, 768], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b7cbe3c9f4e96080d2e34db947bc3016(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ce87c95e94f955d6bbc556fc3f2a3237(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a3ef6794dd2f72a6f5e67e8d6c1d70f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4f3aa206a20021b6e4261eec76a2e430(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f4bdcc37f7f5f5ccbff2b227a508996d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4e7c89f21ff38666b13d49a65c6ac09e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8f75876c0a9a3708fff4fedcbfd08709(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f4a1056a378da59afbd4874db1de413a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f4c581bc1ce039bcfd36b136c1618a0f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_365b2bd9e0c433d4f8cab84dad459975(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4c581bc1ce039bcfd36b136c1618a0f
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 128], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_09e71f7b743385fa9e65f4115e3e3a56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4c581bc1ce039bcfd36b136c1618a0f
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 128], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0d3785f190e39e98fa47ec1e321da0e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 43, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1d2bbd4abe06193cc1d0819287506796(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3785f190e39e98fa47ec1e321da0e4
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3622ad2d6821d7ff97e2edc17384e9f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3785f190e39e98fa47ec1e321da0e4
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ce1a9beb8570e9bd2dbbb7abc7524c2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_05a358f02ef396ac4a848c6d426344b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b2573c48adfa3bcfade0146e313b573a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a7c7011849b61c458b5bf9c6fbd0a843(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ea5c52a37ea3ebdf749f0c7de615e713(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 10, 4, 320, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a9b8894cc42860fbed672be9e6eb3a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea5c52a37ea3ebdf749f0c7de615e713
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_443c251aa64505e3b91a82e3b8226744(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea5c52a37ea3ebdf749f0c7de615e713
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c9e2ee7b72efabdaf656f00a9ce43844(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea5c52a37ea3ebdf749f0c7de615e713
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_74e8f3d22f1def3501aebd347846e17b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 7581, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f8cb8985d8809b879daee439c47dcae3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 7581, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_78e816057fa2cb342868fa1823edff7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 7581, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e25eff27d359a8fbcf32c0b79ec71449(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([528, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f7df89554ac103cd1ee0194dfb5b7fc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([528, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0f647f9de21f3472b987bfe485a30ed8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([528, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a64cc76d23708495e280914f5eb59d27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([528, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0ac0bc6908648f8c0efd39feb3807ce9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 25, 38], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aa8e45ef510c357726957403aa4affc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 25, 38], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_588a622c55742c3fec07b21ac391467f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 43, 16, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6e6ee598a317be2ea2bdc0904dd031a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_588a622c55742c3fec07b21ac391467f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a7958cd199e0de9c1a382a8485779673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_588a622c55742c3fec07b21ac391467f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2eaca0591983d0484d8a64c636a8d859(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_588a622c55742c3fec07b21ac391467f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a20755a6e89498b0977604827ce06d5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 288, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a9b53bf63541a9ca4a053954651fc259(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 288, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4d59ef472a90b9a1468d07661f1a6c53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 288, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c51d319d261f81a112d50ee14ca78c09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 320, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bc1fd1af93e7374bb2492b9e5cd158f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 320, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_586fd7c224f1fe6e9b8194acc4762595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 320, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e65991bba8ca70f0b60e1557f973e820(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4725, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_68c7fdb09ab4ebaba87dbe47acc895ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4725, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_34b9f8e055e1c2b7ef106d9c0e372f5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4725, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_55fe1fed3749416e59fe856e588e74f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_69b401aff239ce65a184435745c45a42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2a437d6fdfd627d9becdc4a365293b20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1c3036b59beacc3df99feb9edb281be3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 577, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8a0e402e2d83ef4bcdcf4e969626e1e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 577, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f6dcbd13e6ac359be84d414b6e28311f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 1, 12, 577, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d52cddee6309ff9cba318e4247f6e673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6dcbd13e6ac359be84d414b6e28311f
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bfd7b8641d1e5f72b87aae070c2e1bab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6dcbd13e6ac359be84d414b6e28311f
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_67f2dbf8fd6d2516b180fa8a79aa419b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6dcbd13e6ac359be84d414b6e28311f
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_191cbcdbcde9f9fbfcff0f8fdfefdace(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9fd67ee0ef546d492204a10acf2b1705(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_136e8f7021d57f44ecaa6b47d6071482(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5372401f5d2f50fa99d5a05becff6d15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([64, 64, 32, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_be385a3364166f9f92ee6ad954976037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([64, 64, 32, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_09bf91260abf0c935e20c815d076f913(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([64, 64, 32, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_af19925a98287eb51ff4c5fbc0232a5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 10, 6, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_166d368e5c3e51817efa5b2a4fb5cc16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af19925a98287eb51ff4c5fbc0232a5b
    def get_inputs(self):
        return [
            paddle.uniform([2, 10, 6, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5a8b915497708de2f28d7f6b1f8881f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af19925a98287eb51ff4c5fbc0232a5b
    def get_inputs(self):
        return [
            paddle.uniform([2, 10, 6, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_61e4cdf9a9dc8a45db6f07608334aedf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([384, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ae2e56e3f19eb2e9688bc2116111fd1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([384, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9eb1ca899e387db8c25bf28384b9abcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([384, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a19582df57dbdd8fb3fe50b97fc23eac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([384, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c57941e5f1f365fafae371701a4b3a0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2377e06c92da7a812d4ab4ed5fff2f26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_427af129532444cfefb9e8747adaaaa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 28, 28, 192], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_924fb76d93eb155871ff97412faef4bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 11, 16, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_da7f13749936e6189ed81d8440d78618(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_924fb76d93eb155871ff97412faef4bb
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b45a9929855d99f008d60b145cc9e60e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_924fb76d93eb155871ff97412faef4bb
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2b0132dc6b3ac23257324da68fb11b9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_924fb76d93eb155871ff97412faef4bb
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ced8d7d22fa3a211bc8fa62ee71bc826(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 2, 1024, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_13c949e28a6b4a754152579174f3ce0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced8d7d22fa3a211bc8fa62ee71bc826
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e3419eec94b9a2ba54ad213f2fe5d7d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced8d7d22fa3a211bc8fa62ee71bc826
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1d2bbd4abe06193cc1d0819287506796(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3785f190e39e98fa47ec1e321da0e4
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3622ad2d6821d7ff97e2edc17384e9f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3785f190e39e98fa47ec1e321da0e4
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f3a7344e738f0c1a947252e9406f43b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f3e0b55fec6cb06166cc7f5462f8f487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7464d073d5480afda5d5ed52b500feb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8c1b948f72b49798cbd5781ec6b0843b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 43, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6e5327819aea278cdedcb3a815420726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c1b948f72b49798cbd5781ec6b0843b
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eb8f2e5ebda59ebb982dc9e612d051b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c1b948f72b49798cbd5781ec6b0843b
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ebfa1e480618a1c25b69791c6c0681a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 8400, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_656e5b2c69ea6df04c47e396ad3bb39e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 8400, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d6503315c8014fe7987e743aaac81745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 8400, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_17fa7b0994fd38fea5d67cc065d5fdb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_43d266db7014b828d8684de56ac08fb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3b62e9b0f9fac945d4448bdde0209d66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a730e156b56e79aabc17ff9565c61be1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_56415c0aed9906237347dc518facb4f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 20, 30], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f4cc5a8a4170fccb839aa4a3cc386de6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 20, 30], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_36540f73782ca61670d73c4b8ce8dbe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c67278e6c9b2ec92dee229432f2f8198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_88264467607048de1abe20ac17ffe6ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f8d57212f315c7d19dc9cee65a7c2b5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2992e4635e1d7d85713908f9e09c0a3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_44fdcba32943ded3019eb64b897e609c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5fef3ef96b7a2728d35f3e376bb95824(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5be4bb1e94f7515b05aee45f9ec7878c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_479de74b9071fb12e4ed646fd6d70867(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0493be90b9f868b61ae062cb6e7a0279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_581fa35d36f3f8d58d8da34f677138bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2b1e1c5cdfe4ebccb95e7192f7b25055(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c08b9aeaaa3a04fc4ada62bf9b9c1bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_755032ca0913231ad3cbb0ad1190f39f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f566ec727c6a6d07457c9e29461b0153(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f72beb0e25895f3b94a526a636dceb1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6efd6f900850e9525f3021e0f9833aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f8cdbf294e842b0aa6dac272dd6682b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f4bdcc37f7f5f5ccbff2b227a508996d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4e7c89f21ff38666b13d49a65c6ac09e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8f75876c0a9a3708fff4fedcbfd08709(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f4a1056a378da59afbd4874db1de413a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_11de2c07417268211ec1f800e237a915(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b30753fd015c74a18dcfe7b805f1fd71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_449271f7029bd6da74f128177f33065e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_419ee6cd420a6bb006e85d23a1db0f7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e8c81928ea99514bf24cfe838ab4402a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a5e8acfcb911a4f2ae64f8e695651a5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_85466815ac33a45b55cf00a744142c6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e335d26b0380a4e17360a54fe57eed72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ca02d8e37b6b957e64ec7ad7a1e91ff0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_795ae4e5d9cf7e901c92fdc1b0c10fdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7f1d83217d5c295dcd1a3d6ea0e3da7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 10, 2, 640, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eb8066f1a4f8866f424452b333666a0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f1d83217d5c295dcd1a3d6ea0e3da7c
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9038452db81b595310d903b88025da65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f1d83217d5c295dcd1a3d6ea0e3da7c
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e91163c3f3c8db90555a66f99fba4bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f1d83217d5c295dcd1a3d6ea0e3da7c
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d4a3952d540796912c52c02b679f8c83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 86, 3, 198, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c2f598014b401942eb53fbe035f53bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4a3952d540796912c52c02b679f8c83
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c9c4a0ab8d0b2c1e09c0d3976fe5c2ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4a3952d540796912c52c02b679f8c83
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_85454ce419fe562cf38539edb75f4089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4a3952d540796912c52c02b679f8c83
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4ecb9138f0021b6edb980298b58d8397(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0238a9b7251f0d40866cdf80199c02f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2be559ce424338711c1cbec0c7d16111(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 11, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_46003d221ae61bfb4c9473cc2a7e3592(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2be559ce424338711c1cbec0c7d16111
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_35ac55a4bc33a4965d894bc6c2505ce8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2be559ce424338711c1cbec0c7d16111
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_817ba0e3a5f2055229ea8ff37315a847(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d4660134e83fa3d5f55e07b75cd2c721(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_94ba48ffe7afda4287e049ad04d73ef4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2b787bdf7cabc5e0e2a0d90433919525(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d7f47fac90b5fce39ba5acf7e3419fba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([20, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ff5cacdad07ec3628e72e64e1b416921(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([20, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a16366026d170175a09a3b9126388b78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([20, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fdc96ffd33c3283e47f31a25797a1791(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([20, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_31c5f542173731900e3b7b45bd837f0f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 43, 1, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e51c8ef971ddfd6c79fc9376ef5aeaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31c5f542173731900e3b7b45bd837f0f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4f4162fcebc2f1f74f46299cab2ca6fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31c5f542173731900e3b7b45bd837f0f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_de002ba22d4f4c0892885a7d26f0ff7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31c5f542173731900e3b7b45bd837f0f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf62292005b79a8147f8331cd3e29466(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 97, 97], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b2613ed05de5cf8c3ccaeedfdb239e6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 97, 97], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7d730f3f84733a3fbc0c12095f663d2e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [-1, -1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 104, 104], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d0e11e6c2a88e97ae41ee18a03a6c398(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d730f3f84733a3fbc0c12095f663d2e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
            paddle.to_tensor([98, 99], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_4bfb6d392558691a4598f8521a6a42d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 4312, 4, 16, 6], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a4baa35591bef4a59546bc55cace603(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bfb6d392558691a4598f8521a6a42d3
    def get_inputs(self):
        return [
            paddle.uniform([2, 4312, 4, 16, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d77b1c0c8d875c52b584dd4a37d9e29a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bfb6d392558691a4598f8521a6a42d3
    def get_inputs(self):
        return [
            paddle.uniform([2, 4312, 4, 16, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_123390e893014beda794837cbe8ff1ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fcd17fa1e46e787b3f84ee62e6cd2b6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8ee9f884eca2c6e32fcb5c82ebaf6ffd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f063157bd604da0ae2ae05c617410ad
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a8a44decdee11e8266da42f588050149(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f063157bd604da0ae2ae05c617410ad
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cc5069a45751f63b344845c741cb95ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f063157bd604da0ae2ae05c617410ad
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_da1db44ad577373d5b27f1818e51ae3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e6c3a5eacae3efa656221d8b218b6b12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_75ad94c70bddde84e694450addca52c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 7, 7, 768], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_79bf2e77e94078019e74ff6167703e6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 1024, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_67dc296f2f910624653145748149cbd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 1024, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0d66543c406954ceb9982ccc6b6ef7a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 1024, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5df0f96980e66b79620487adb1c302aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 256, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dcc8afad1b52135ebba4ab44968d7699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 256, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0e3f7e9c9312e92c220ddefeb7d26373(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 256, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_29173c18ee6299f2a13a13e4e051fd57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([576, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7cf20a4890f49aa19ce5f8e9d60523e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([576, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_614e50b1a690955445e06d8a3b52989d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([576, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_28f185f02c5187b7583266c450e8e624(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([576, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b9f784e1cfba26627f33127f6b9e01ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([22, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d3992d734700bb2022a666b8ad02f0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([22, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c93175243051bdfa43b355b3983309a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([22, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dcaab71366ef07c47fb41133f43deaac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([22, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_58584606027cfc602a1e89255244891a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([96, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9909d64cca8d8a38415173f9a5650d17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([96, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9f4905de0ad74f55cbd8f7b821dc30ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([96, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_248cf41d35a4ea0c4c655277e3eea109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([96, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f4ddc389e83695dc81919bd3706b7764(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f1474cb37eb462344219c7b7ca225b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47103e6e4f8774ac41e27b3a587befdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b5e3b22a56a9c190d4c7ef289916c7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_771991a7aae417400684991c43f5ac51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 256], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b5eb489206548cc11dbc76eef7af4982(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 256], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4e5312de63c3d4eff8a9cc729eb298ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 8, 512, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d543985344f1fb8cee81b7a0b06b6e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5312de63c3d4eff8a9cc729eb298ae
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2aa6eb6f7e45073ce43614287dbfd493(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5312de63c3d4eff8a9cc729eb298ae
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ca02d8e37b6b957e64ec7ad7a1e91ff0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_795ae4e5d9cf7e901c92fdc1b0c10fdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_614ae0e70aad2f5268f8504f4dfd0d21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_011f2628076a232cef3be9fcaf08ea1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_678284f1d93061de653c1c5d147b4cbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_424797ea8a71792913ed3563c494f454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_724ff24531c0eaec3753e32186df779f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 11, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a36d25e80b4aa33c8f3cf5172a8b9751(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_724ff24531c0eaec3753e32186df779f
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_887edbefdb60b0485a16a436f6341b89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_724ff24531c0eaec3753e32186df779f
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8465d6d301f18a46c2263cace76b05f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([960, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1b48463798e09ee375c78694670fae3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([960, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d98ef6d4351fb39d515c52fa7aeac3aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([960, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04d0b4d182e613a56002231c918af56f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([960, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b3ad386b33b73b61f841b1c98e9c30f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b625734245499b5ce7e9f95a6ca197fa
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ef9101d31f213bd62f7186855d0134dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b625734245499b5ce7e9f95a6ca197fa
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e67c9d49e6651db4cadd688757b70835(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b625734245499b5ce7e9f95a6ca197fa
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b466dc1071346547135dd43b4655f573(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b72c8a8cc4a0eff57d71cf709c61b656(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ccbc1decc6660609ba7ad8594bf7d59d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 256, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7cee05b50945d69f7f7ff92a0d64da2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 256, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1b59029e8a1bddb7b4610c65a3f7ca0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([2112, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_af97c714e150e74a3c22ca8c095c06a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([2112, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0b1206f5eae8ea7b00ada750ad1cd456(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([2112, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b406b030ae68d6ca6c8ac5a60065fb73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([2112, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_62a1c4bd227661492a2ff2eb224fb8a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 72, 28, 50], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c0a9673781f326a0c7615d760c31d2e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 72, 28, 50], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bb184811b825a6abbc89a3e1fccfd96d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 72, 28, 50], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c4fbfb7d91ede0eb6ff2f9012610b6ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 72, 28, 50], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c6e13070b5526c5523468abbd53708a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a25656aa6082221b78367e1f8359591a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3385549485683441, 0.25113946199417114, 0.2512962520122528, 0.20602960884571075]], [[0.38148048520088196, 0.15045665204524994, 0.32663360238075256, 0.246501162648201]], [[0.25288909673690796, 0.13170458376407623, 0.14401888847351074, 0.29534777998924255]], [[0.03507382422685623, 0.23587974905967712, 0.26744359731674194, 0.15506257116794586]], [[0.24599219858646393, 0.01184688601642847, 0.37740594148635864, 0.06887172907590866]], [[0.05196724086999893, 0.42191997170448303, 0.059061601758003235, 0.20347267389297485]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8114606c760ca49c86ba523193bbdedd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a25656aa6082221b78367e1f8359591a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3385549485683441, 0.25113946199417114, 0.2512962520122528, 0.20602960884571075]], [[0.38148048520088196, 0.15045665204524994, 0.32663360238075256, 0.246501162648201]], [[0.25288909673690796, 0.13170458376407623, 0.14401888847351074, 0.29534777998924255]], [[0.03507382422685623, 0.23587974905967712, 0.26744359731674194, 0.15506257116794586]], [[0.24599219858646393, 0.01184688601642847, 0.37740594148635864, 0.06887172907590866]], [[0.05196724086999893, 0.42191997170448303, 0.059061601758003235, 0.20347267389297485]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a2495438a5302ea3a1013ae7ef33ab91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a25656aa6082221b78367e1f8359591a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3385549485683441, 0.25113946199417114, 0.2512962520122528, 0.20602960884571075]], [[0.38148048520088196, 0.15045665204524994, 0.32663360238075256, 0.246501162648201]], [[0.25288909673690796, 0.13170458376407623, 0.14401888847351074, 0.29534777998924255]], [[0.03507382422685623, 0.23587974905967712, 0.26744359731674194, 0.15506257116794586]], [[0.24599219858646393, 0.01184688601642847, 0.37740594148635864, 0.06887172907590866]], [[0.05196724086999893, 0.42191997170448303, 0.059061601758003235, 0.20347267389297485]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c64d33e3a55a7cd76d4929b1c0035258(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a25656aa6082221b78367e1f8359591a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3385549485683441, 0.25113946199417114, 0.2512962520122528, 0.20602960884571075]], [[0.38148048520088196, 0.15045665204524994, 0.32663360238075256, 0.246501162648201]], [[0.25288909673690796, 0.13170458376407623, 0.14401888847351074, 0.29534777998924255]], [[0.03507382422685623, 0.23587974905967712, 0.26744359731674194, 0.15506257116794586]], [[0.24599219858646393, 0.01184688601642847, 0.37740594148635864, 0.06887172907590866]], [[0.05196724086999893, 0.42191997170448303, 0.059061601758003235, 0.20347267389297485]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d732d560c5e1c7dc3b8b90c4ec4d9f8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4116, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f3097b77d9d8273aeec80502769edb17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4116, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_77d79c36c4c7e75d4502fc167d356bda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4116, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_859614572eb1d4eeaa9f578d68ed3c1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ea2b075f277df0f9769c469036c0260f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bbe0fbf7bef637c0503438e881692a81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8a1e6c9ed6af8300a971fc24e5983b3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5a507605cb2fe23faa836fecc9986cda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6d64419b102065dc4c8ac7a0369cb2bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_22c396745b61d9565a7aa485d53ce7f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a1a1d34087d439c6131d0bbb914f5373(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_30915a79b697c9a907fcd61b2a522e8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_83757eaf8a1958b1f18ac2b194b3a5d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bb63f61dbb3ab4aef55c5d93cc054460(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9c28039a46df08c9f5610e00e6b2548a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b2ecb488b26e1c28cb6c253359c79407(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_45a744f0d83b9a4333afc1ae3a9fac35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d03a65306d98360e10d8f0be26d1b5b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4ef0c393bf56d46e0e09e7a2f9371fd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2ab4958eed1b84725d84464c757ab440(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_14d33b6c76fced2eab2904e6d15e6588(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_91c7368d876500547acbc9d9ec84b9db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bff1bb2320892359be6113a7c679634f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_171c7d96cf5b0a719367f5b602948936(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_16c83f22a2926a2bc735d81a0c0d3504(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_831a74fd32fbc2fa4f7d82de8e567d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b43a6e90cd84ca42a05b7a367b579574(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0bbeefab5147e756c78a696361b074f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f70e4d67c9737c06b081fdb41d42bc59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0a93e20eff5c1babc06bba84d2586036(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ecd0515369925e60836c6ef7fca8f438(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3b6c7fa7973f594d0e51e0ff3dfe9e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1ecf5ddd99228049305f68e1a0610b95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_90863e5b3d049ece7f91c2a146be63f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dfb4055d9a160e7a71c10f86f253e113(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0d54744d954ecdbf30a8c0c98b987885(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a2d644acd91977ae5ddd5422f7a53e49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a4da53931658b16c5f216b6866f7cb7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_06097d37e0600d20b0ef7689dc2df77c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd2cb3616da57872beee4c6f016cd5e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3e83d730d87ff5b8243e119df1c067c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e0502ae7628e5f40813e601b896b1d18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fd598bd28598852bb9c7bdee276f7fc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_59ba985c4b8b1475fdcd8830403af8fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ff0f2d7eb833ad211e0a1b682dfc75d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_16c06e1918c29a529438ac3aa2a3998a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7c7b4bae51a7c155905431c3f837ef9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d73b98d9e64e911d3708d266f05f9fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_456103403280f121f55367ae7aef50a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cee4b4bd545cd08585ee0ffa37abb014(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_66c61d579b07b796eda7ad55122036c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3ed31c00e260650e034991a86d58fbc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9f2ae29ac07dbf63111cac2cbc40b855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_26178ee8ebfd8bab8caccdae29b68892(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_859614572eb1d4eeaa9f578d68ed3c1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ea2b075f277df0f9769c469036c0260f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_407751eafe64de46ff217b3176b5df93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b6487646d4fcbb3928720078110cbbf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f4bdcc37f7f5f5ccbff2b227a508996d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4e7c89f21ff38666b13d49a65c6ac09e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8f75876c0a9a3708fff4fedcbfd08709(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f4a1056a378da59afbd4874db1de413a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2992e4635e1d7d85713908f9e09c0a3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_44fdcba32943ded3019eb64b897e609c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5fef3ef96b7a2728d35f3e376bb95824(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1eab0a90e1d7cdd2422327230440e074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 960, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b54fff3900946ae2ea5da3f778525204(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 960, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7bc2e9ba1e8aafcdb63a2c597fbbb6a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 960, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_96de69347b9da8b175fafc7ece8687bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 960, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_592d02e7e864ec125787835f77f1c16b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 480, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ad273715caac112b52417827ec4e9265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 480, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4db1bfe813412e735b53c1df8638c7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 480, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_33507595266188c7110efea07968df9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 480, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7769fb2de264a6633758252f7bff579f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 240, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f47ef71b178f97ebc985b54547571dab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 240, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f6c668d5f3b1247aba71f1076a0c8e37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 240, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_41725b6a8c2ec32174364aa7cd9c6519(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 240, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_27bb7f4518e91a51a475bc7d75dba484(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 6069, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b93e52b4dcdbb9a52fe44d600d58ba15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 6069, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4e12a2c399926cde92d44394ca5ac137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 6069, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1bde50d097be4731d7df3a0607bef7b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_607bd232de8f00d148e6e83e04e0103a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2826668fd87b02f260f0f522b830b8d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 28, 28, 192], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_40efb01fc3c3a1d1ed0baf65103802e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_da7f13749936e6189ed81d8440d78618(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_924fb76d93eb155871ff97412faef4bb
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b45a9929855d99f008d60b145cc9e60e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_924fb76d93eb155871ff97412faef4bb
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2b0132dc6b3ac23257324da68fb11b9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_924fb76d93eb155871ff97412faef4bb
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ddd62463173e17d71dd92d6dac2b0d3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4c581bc1ce039bcfd36b136c1618a0f
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 8], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_871a22313c75a354b6e109363853a9cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a175d48813094269c35116a220b3591f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1f6b5a569321115d4e4e31e9cdfc2a49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 1, 6, 1025, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e591a77493235029cd5fa36e823cabbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f6b5a569321115d4e4e31e9cdfc2a49
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_52c71a8221abcd59fce6bc6d196352ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f6b5a569321115d4e4e31e9cdfc2a49
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b6a8715882caa2af6694caf1d8105282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f6b5a569321115d4e4e31e9cdfc2a49
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cdee69ab42b9d16e1d0f94dcb1688c84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0ce44129f0b5564a009657f1641918c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e4e5ada49ab5ada7a2ac34e7b3f8e84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d65cb757f437e66e0d26fec18be8393c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_537f55299002dd3ac2192420621fc1bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_386b4133389b0f09b0838251ea1be469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c3563a5c2ac78a50f521cfa67eff89a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_fd2c04d2f4825e7ad9ea72ac7923493d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 11, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ab3c51f3e6dfda6e9ac753f495c745e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2c04d2f4825e7ad9ea72ac7923493d
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e52a7e8bab3e5f06dea7a5d4c2769481(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2c04d2f4825e7ad9ea72ac7923493d
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf14a27b737efaa7f619711cb0726ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4c581bc1ce039bcfd36b136c1618a0f
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f61c1028bb2c3b198fe18c005eb3cbc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4c581bc1ce039bcfd36b136c1618a0f
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bf0731a9e497771be1ff8aa121e3cb3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_40efb01fc3c3a1d1ed0baf65103802e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_191cbcdbcde9f9fbfcff0f8fdfefdace(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9fd67ee0ef546d492204a10acf2b1705(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_136e8f7021d57f44ecaa6b47d6071482(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_386a1c5dab52ea2f3d195ce07e0239f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4c581bc1ce039bcfd36b136c1618a0f
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_764fcd3233986fbec07886ec2c5bb789(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4c581bc1ce039bcfd36b136c1618a0f
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_abdc5913fec9093f24759a38d75ae220(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 5, 512, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_908ba4063d5271da1467f387147b9e5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abdc5913fec9093f24759a38d75ae220
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fb5ba9e0450ddffa2e93b0574a47ccbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abdc5913fec9093f24759a38d75ae220
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5e284aae983ede33fc33fe542408b868(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cbb35bd16dfa5d89e64989c269c98077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e284aae983ede33fc33fe542408b868
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 7], dtype='int64').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f61f68d05cd9bce6c42aa7f909af9f9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e284aae983ede33fc33fe542408b868
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 7], dtype='int64').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4db497ad26750eb96b477e1cbcbb05c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 256], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_265892c2a95fcd0fb3d8479c57d6efa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 256], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c7b972fb687cfdc1fd8843cf56a7c6bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 8, 1024, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07dde647a236d78ad8dd9d01ac90430e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7b972fb687cfdc1fd8843cf56a7c6bb
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_698edf2925d8081c077684efa2f69802(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7b972fb687cfdc1fd8843cf56a7c6bb
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_322906529eb1fc3790b91d1afe20a48a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2854fed0a198c01a0cdd9839059c8a8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ad824ac59cf89743e56451dfce4197dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_be6f74564f94616fd9b3f8822846cefb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8f87f669abcdb86c1b01768647cb8fe
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_554beb005b034e8d261f1b8cb829dfce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8f87f669abcdb86c1b01768647cb8fe
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4d5bd3b3b1ab4888f8aff4bfb06fcbbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 464, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ea88ac8d0c0b70d8545d8cc60c567522(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 464, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_731de8d5a9df1e1e1bc3e59c846cad3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 464, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_059acabd0427cb34fa1c1ff9ab22dc1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 464, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5e51c8ef971ddfd6c79fc9376ef5aeaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31c5f542173731900e3b7b45bd837f0f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4f4162fcebc2f1f74f46299cab2ca6fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31c5f542173731900e3b7b45bd837f0f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_de002ba22d4f4c0892885a7d26f0ff7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31c5f542173731900e3b7b45bd837f0f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d5b8a90f0a4c949cba9761411a8fa6b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 54, 3, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d4268b253c07c451ac8045b41296766(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5b8a90f0a4c949cba9761411a8fa6b2
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_033cdc032f6a13985aa629189aaed3d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5b8a90f0a4c949cba9761411a8fa6b2
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_55d2bad54c75f9a8835a1cdebbb92ca2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5b8a90f0a4c949cba9761411a8fa6b2
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f566ec727c6a6d07457c9e29461b0153(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f72beb0e25895f3b94a526a636dceb1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6efd6f900850e9525f3021e0f9833aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f8cdbf294e842b0aa6dac272dd6682b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f4bdcc37f7f5f5ccbff2b227a508996d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8f75876c0a9a3708fff4fedcbfd08709(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f4a1056a378da59afbd4874db1de413a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f566ec727c6a6d07457c9e29461b0153(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6efd6f900850e9525f3021e0f9833aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f8cdbf294e842b0aa6dac272dd6682b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_08dddf6e35aeb19f9174863232060dfe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 1, 1024, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_520fe3ca5b8a08ddef577ca020cf188f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08dddf6e35aeb19f9174863232060dfe
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1e0269246ac30461367f35c0b74ad0b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08dddf6e35aeb19f9174863232060dfe
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_579c7c9c1a8d0112d0b63ac9388217a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1f412f03d27b0c1f386d736846ac8d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6e6ee598a317be2ea2bdc0904dd031a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_588a622c55742c3fec07b21ac391467f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a7958cd199e0de9c1a382a8485779673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_588a622c55742c3fec07b21ac391467f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2eaca0591983d0484d8a64c636a8d859(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_588a622c55742c3fec07b21ac391467f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b258a97ab1e50b0c7ef5d5b4a91e7d12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_531b1ee814811dca5ad95282e4e4a425(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0d63e69dcecde129c5e449405e8de739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15a8e74afae0aa29c79cf73760e7363c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f91f306a5b16821fc31d864318d432b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7d476e0bb9c6517bf217eacc5e59370e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9b4c06dd8ad42550b3313a3a2759ce5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8a6c69c9753362506521c5d2ea66cf3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0e1993876db4f3dd3a18441093e8fe51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9dcdecfff7340a78d8f3c6fa6b0b71a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0ba2b3befd0b75a1d4732a9548917316(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 14, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_389acbb6d37a7ba20bf3ed95b697c006(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 14, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d4987fb82517d51fb3f0eb7d6fdda61e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 14, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e12a42d9d84412a078caefbbc77820b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 14, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_46003d221ae61bfb4c9473cc2a7e3592(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2be559ce424338711c1cbec0c7d16111
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_35ac55a4bc33a4965d894bc6c2505ce8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2be559ce424338711c1cbec0c7d16111
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2fe275b4048361f7344dde0fe956c04a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 128, 1, 1], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7f631f62cf746f55b5c4ff6f4b25438a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 128, 1, 1], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ca86975b97ab3e76fae5640fc761a45c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_812d4312316eca05de33a19563d7f7c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5f00a27711b55e2c6c6732fabb722747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a9b8894cc42860fbed672be9e6eb3a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea5c52a37ea3ebdf749f0c7de615e713
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_443c251aa64505e3b91a82e3b8226744(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea5c52a37ea3ebdf749f0c7de615e713
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c9e2ee7b72efabdaf656f00a9ce43844(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea5c52a37ea3ebdf749f0c7de615e713
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_701b80c28961874eab9a631da53dce42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 1, 512, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_133dc229fac1bd18b982dd5af3c199f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_701b80c28961874eab9a631da53dce42
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6759b834a5150107ac26ab537d6e1495(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_701b80c28961874eab9a631da53dce42
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7ab3c51f3e6dfda6e9ac753f495c745e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2c04d2f4825e7ad9ea72ac7923493d
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e52a7e8bab3e5f06dea7a5d4c2769481(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2c04d2f4825e7ad9ea72ac7923493d
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf14a27b737efaa7f619711cb0726ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4c581bc1ce039bcfd36b136c1618a0f
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f61c1028bb2c3b198fe18c005eb3cbc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4c581bc1ce039bcfd36b136c1618a0f
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d8ea3acb02494632235f86136968867d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 10, 2, 200, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca33b62a8da9da981fb923330d429bb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8ea3acb02494632235f86136968867d
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c3a4f11a67490a97d9ee7c888fd02ed3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8ea3acb02494632235f86136968867d
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_65cbd2d667a72baf4e67900a31db2781(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8ea3acb02494632235f86136968867d
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c83b40a8077b92199d9f82333c714a67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 9261, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_22978a90399b337cb8e6c03e642023d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 9261, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5c8ba535d12b4895739d41c9f6f2dfef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 9261, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02b4d637b0ae08a387b01e9bdfa8b76e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aed3a98e1d4f83137fd2afc4c9407b3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_58826191ac96ecf3c5d47027d97d4e01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a6facc94f5b0b543fa611737fae89f18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0363be3d18eabada4bbc1b9a2e25cb46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_67ae8d968c7500d8873c09ea224f2ebc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bc6c49d773753b00a8c9cd896a283fa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3b7ebe96a4bbeb8c81d20cb7d5111a53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6869d848606087e08c2a10991e54a9d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e475d48caf3cbd2c0afe8ebc8eced659(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_df76951a67d186a5ea047d9fa4263aa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_401ad0240274c42ca6e26292a9dd623b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_df6f53c951dd37a5f1160156c4e63c57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aae645a4b77cc6d7b1ed30cdebee5227(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5716ee822b3dd4b9f49dc5b064b7bfaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9a4e9cfc120401e3145783ff26267322(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_24022047511fdf364e3597cab7b65371(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 22, 6, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_64663cdf2f4b502669c117bae0ce220d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24022047511fdf364e3597cab7b65371
    def get_inputs(self):
        return [
            paddle.uniform([2, 22, 6, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f5bc3c534d8f28178b7b35de500fcbd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24022047511fdf364e3597cab7b65371
    def get_inputs(self):
        return [
            paddle.uniform([2, 22, 6, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_88631fa06ccaed4ec33baeb7a1233a30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([240, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4abab7fb5a0725cdc481ce39a40c9d23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([240, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_31de4e16e1e33c29b9fc87035c4550e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([240, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ae5bb815930e425bffeb4a3f3dcd3b89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([240, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ee17cb6468439ba87b30333c9950cf04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_30e9ec8bb562e15299ff217d327b0835(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e2f439941138093b5ddff4cad7510ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b2ea5973c8d4f7529863e89e67cf6f64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4ecb9138f0021b6edb980298b58d8397(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0238a9b7251f0d40866cdf80199c02f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4ecb9138f0021b6edb980298b58d8397(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0238a9b7251f0d40866cdf80199c02f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4ecb9138f0021b6edb980298b58d8397(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0238a9b7251f0d40866cdf80199c02f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d15db0d69c3343745a59359c3b2f9cd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b7775f69339549810d66f4612cc265df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1bde50d097be4731d7df3a0607bef7b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_607bd232de8f00d148e6e83e04e0103a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2826668fd87b02f260f0f522b830b8d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 28, 28, 192], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d55aaaa33adefb17f34b84c755d178f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 43, 4, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9311fcc082a1b7ce412c28b72b48c318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d55aaaa33adefb17f34b84c755d178f9
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5d271813a4ac4ec78905f3199a1661fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d55aaaa33adefb17f34b84c755d178f9
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f8527b2aa20d9dedd0e25d05b9dfdca0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d55aaaa33adefb17f34b84c755d178f9
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_375cab641a97d0e4aa15d5b2c6e1b112(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 2, 512, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9913c9155206c49f2a7ba4a56e23f9c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_375cab641a97d0e4aa15d5b2c6e1b112
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e110870a4bfb54ed62479805896a84e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_375cab641a97d0e4aa15d5b2c6e1b112
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_da2deca74b622bd97753b5fd7fe459d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e362a3f2574f77f14ee517eec775c77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d2658f5cb6868687f045c5285e8ee445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3396acb88d1ec76f666bbd7cb7c672c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_db37ee58672f6b1d44a4ee9bd0d79e52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8423b9183a7d5ac6f7b7a95f9e904e0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d56771eb37a1d11a8e32d78dbc2a3b1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b96f7b3e6e33974a6b268f5b019ea545(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0d6bb7c44480be60732621ac08ef6940(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_050b00a64310d6602a1aea7815ce911e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cbf7027b7f624ceccfb1c2e23f3bd70b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1db07f6287560072fb2d7c4e0bcfd623(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_18bd62bf4348e43f8a08c5a0bc72afc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 16, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b9782d6f598c4c3092b594cd18204656(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 16, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_35eb11b5376ab9e4e160ea9d3733347d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 16, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5b43d957f78bc2d41f17ebe66ae74b35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 16, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fa1ab46f63a81ac95691bdce23276150(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 8, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f0fb3bcb8c3edfd0a2daed8c135f30f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 8, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_52578fa564deb08671def8224ee2e4da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 8, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_20e37a21aa6ffe57a9b05d99a5ccbabe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 8, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_954c27ad9d9560537cec53757b48deb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 5, 512, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e2f57c90e8a4bd97de60772486020357(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_954c27ad9d9560537cec53757b48deb7
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_22bf8e02e26479140ea4b390995d9b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_954c27ad9d9560537cec53757b48deb7
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a573ed73a07b43d303da6f34f63e538c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2100, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0c13839eb9b5a75d910c97c9be881c77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2100, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0bf7aafa26363d32d4c897c0e6ca6b1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2100, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4ecb9138f0021b6edb980298b58d8397(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_740383d17db3909973ea449e25e84bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1d2cd4615df4a078698ebd9344608d23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cdee69ab42b9d16e1d0f94dcb1688c84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_14d53f860f9464d92eca41c6877197e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cdee69ab42b9d16e1d0f94dcb1688c84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_14d53f860f9464d92eca41c6877197e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cdee69ab42b9d16e1d0f94dcb1688c84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_14d53f860f9464d92eca41c6877197e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4a294e62881976eb8e360468c5709a7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2b2c86e7ebca68c0c163aa4f5ceb88c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15a8e74afae0aa29c79cf73760e7363c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f91f306a5b16821fc31d864318d432b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7d476e0bb9c6517bf217eacc5e59370e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d78ef8b86d3edc2b51671d483b360af5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 15, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4f45ab8998c0fb41b26ba3917d9343ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 15, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_191cbcdbcde9f9fbfcff0f8fdfefdace(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9fd67ee0ef546d492204a10acf2b1705(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_136e8f7021d57f44ecaa6b47d6071482(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e1b2bf71982cdba2694a56be835df15a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 11, 1, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c00438f5a5fd8f74070484dd06c46db3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b2bf71982cdba2694a56be835df15a
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84c10549997166b2ebbca38eddbc3cb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b2bf71982cdba2694a56be835df15a
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_423d641cee479523296f4352d381d27f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b2bf71982cdba2694a56be835df15a
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ff1b21f6c7c7c514facff0a4cd1b5e88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 11, 4, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4315ff1176904e22adadb01e907a0b03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff1b21f6c7c7c514facff0a4cd1b5e88
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ff98e614f01a38d1a837f4f5264fcc89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff1b21f6c7c7c514facff0a4cd1b5e88
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_96983037a28bba481fc24e44605129b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff1b21f6c7c7c514facff0a4cd1b5e88
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c00438f5a5fd8f74070484dd06c46db3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b2bf71982cdba2694a56be835df15a
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84c10549997166b2ebbca38eddbc3cb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b2bf71982cdba2694a56be835df15a
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_423d641cee479523296f4352d381d27f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b2bf71982cdba2694a56be835df15a
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_85466815ac33a45b55cf00a744142c6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e335d26b0380a4e17360a54fe57eed72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b7bc3b3fdd25f0c0b14e76a9615bcca4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 1, 12, 1025, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc7fc80cf7c43ea87978390d78875944(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bc3b3fdd25f0c0b14e76a9615bcca4
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2b8bc97f2bcce98c6e34a178c31e4d21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bc3b3fdd25f0c0b14e76a9615bcca4
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6c9b773fec0679ea655ac235317f8f89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bc3b3fdd25f0c0b14e76a9615bcca4
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c01b7948523f02f7dd198ba69e7cabf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([44, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4fddabea7ea06c100c4897b10c7ad6ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([44, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1f9ff1407c68f08aae734a31c9523a5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([44, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1e567f6d8e86f594f296fdd93ba929a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([44, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4315ff1176904e22adadb01e907a0b03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff1b21f6c7c7c514facff0a4cd1b5e88
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ff98e614f01a38d1a837f4f5264fcc89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff1b21f6c7c7c514facff0a4cd1b5e88
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_96983037a28bba481fc24e44605129b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff1b21f6c7c7c514facff0a4cd1b5e88
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9d7d14d0aa731ac0b6422cb4c0afcfdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b72780a3bf834841bbc07cce9aaa6c4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9fc941d2920c1564bc6e8d6794aaa35f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_01ca258e4142b65104c637763da17439(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ca86975b97ab3e76fae5640fc761a45c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_812d4312316eca05de33a19563d7f7c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5f00a27711b55e2c6c6732fabb722747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0415765b2174cc5293ec150cc550160a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fb1a0239b0d9299a9e0cc4a8c2e7cff
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4ecb9138f0021b6edb980298b58d8397(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0238a9b7251f0d40866cdf80199c02f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_740383d17db3909973ea449e25e84bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1d2cd4615df4a078698ebd9344608d23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3df2e71c7ea99cd886e8e6dbe0014afb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 512], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5b03afa6069f32a4638fb520f0d668c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 512], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8d7952889068d6bd7d3c0aa359aeaf3f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 8, 1024, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b4c65bc96be838181c55d420957d16b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d7952889068d6bd7d3c0aa359aeaf3f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ed7d0bb8bc15778d2eb8db3099ae0f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d7952889068d6bd7d3c0aa359aeaf3f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5bd27f46fb312c7709af30a562356233(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 11109, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_57bba4efd6c30cd446684a92c952aeeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 11109, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9d36ae47385a9ba36f1aadcf4f1be5a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 11109, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b2dfab79e7c5d696f69db561f6f97d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123f4206f30e0c7376b3076b177cb920
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_593713920e5f789036f4ed4537953915(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123f4206f30e0c7376b3076b177cb920
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_db12eccaf083484153a4cc1dd1486aa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123f4206f30e0c7376b3076b177cb920
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c5ed2d77a2b0ce54ac26ddbc332e6996(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2641b71fe698c28e497ab88cc823d4d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d0133439bdec1d8ca77d0e9764f05d35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 7, 7, 768], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9311fcc082a1b7ce412c28b72b48c318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d55aaaa33adefb17f34b84c755d178f9
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5d271813a4ac4ec78905f3199a1661fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d55aaaa33adefb17f34b84c755d178f9
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f8527b2aa20d9dedd0e25d05b9dfdca0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d55aaaa33adefb17f34b84c755d178f9
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_da1db44ad577373d5b27f1818e51ae3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e6c3a5eacae3efa656221d8b218b6b12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_75ad94c70bddde84e694450addca52c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 7, 7, 768], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6e5327819aea278cdedcb3a815420726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c1b948f72b49798cbd5781ec6b0843b
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eb8f2e5ebda59ebb982dc9e612d051b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c1b948f72b49798cbd5781ec6b0843b
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f5560710b9ffa50e7baaf4b1dcfb6b1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2fe184c3c7f3b493a1d1ceb6d5011ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_db37ee58672f6b1d44a4ee9bd0d79e52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8423b9183a7d5ac6f7b7a95f9e904e0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8f8157390b5e50b848d448d1077f6647(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([144, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1edc8ee4f859b1899422bd4ab0fbc889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([144, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_63204fe0b8cfb5baeee55c729362b6f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([144, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_abad30c38bd82847367fb0668e2a129c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([144, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ee29d133f50a17e4f8b7a7b66aef237a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 5, 1024, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ec974134029295619680e408deb35cb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee29d133f50a17e4f8b7a7b66aef237a
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a3c08533241e54e31eabafa5b53e318d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee29d133f50a17e4f8b7a7b66aef237a
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_cdfd4675ead612933c750a321127a618(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 5, 1024, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f23137451afae18dff2e198d908b472b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdfd4675ead612933c750a321127a618
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a0ee46b6144575460cf047c5e582103c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdfd4675ead612933c750a321127a618
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2e74110f99ed1185ec21c5ffa288e990(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_27e9dc5646722f4a7d4e7b42c4e56ff9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4b150b07f8df9e3c381bae373a0bb54c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6992aaabc2c028932535401b2fc8c563(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6351370c107ca3bf22f9d660f220a57e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ad23d257a30c5cc7ee3f9aaf8fb81ef2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cd2f85d246eb5e988d48a9b1894a4e7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_36a44245746314e0c457e0ea4a8d578c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_417b619174ff1d18413374a14b3d1bcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 160, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dc234b2c1423e7a70d79b716c460cf5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 160, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5fdf989774ccf92c503984a87b634572(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 160, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1a345689f9f8206bd21cd2a248276ca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 160, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3740ba47233d4b6d84e1d6e4f20e1746(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5eb36276ff4edea9a2dbd6de7c5c74af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b58d1aba6fe5982bce653c0c1117337d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_df5b4e22238c784ec9c76fff4f64fc33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_123390e893014beda794837cbe8ff1ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fcd17fa1e46e787b3f84ee62e6cd2b6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3a76c501a3d440f04f12697f4dbdbbe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2fcc6512c43d65bbf2a2907b50f1835c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3a364ec4023d021275dcf0a44f6f62ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d9cdad4809f77a6a65130780f7ae244c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_336bc09870a73084e0da130cd8f48c80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b6d14c7c0321f81352a9c15c55dbc30f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_df3c151fec3377785fa1e1a6d2ff2ade(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_829887b03bde77c51b681ba316d3074c
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_65a25d64388215630a00a9932e70d10c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_829887b03bde77c51b681ba316d3074c
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_14ed7b3d5f407e1c640b9400925c6200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 512], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8ed01572bdb0dd374fd0499754edb3d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 512], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8c5599eeb84e7d1c890c0a7ea8664030(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 8, 512, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cfe125395fece830afeb78701845b2a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c5599eeb84e7d1c890c0a7ea8664030
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d484a08349fa806a2d4db79cdb3dbff7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c5599eeb84e7d1c890c0a7ea8664030
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c57941e5f1f365fafae371701a4b3a0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2377e06c92da7a812d4ab4ed5fff2f26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_427af129532444cfefb9e8747adaaaa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 28, 28, 192], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_24c998adbc4c938b976241e834620bee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b9e92c45462ace15c3fe73738ae93d24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3c3f3473be1a4ae1c87ddaa779f06640(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cba436843473132c18085e1a14d5ef35
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_17b6d771a7c41c1b551198fee689e039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cba436843473132c18085e1a14d5ef35
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e610d10de835a04ec3ac60fdf940d5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 232, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_92a0b1aacbea417a8429c0705b425aaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 232, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e508a20deabcc9ba84084f1927475fc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 232, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ccbaa5502ca063377238ae30c2dd9897(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 232, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a36d25e80b4aa33c8f3cf5172a8b9751(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_724ff24531c0eaec3753e32186df779f
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_887edbefdb60b0485a16a436f6341b89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_724ff24531c0eaec3753e32186df779f
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2992e4635e1d7d85713908f9e09c0a3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_44fdcba32943ded3019eb64b897e609c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5fef3ef96b7a2728d35f3e376bb95824(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_241e61a5e9e1e330d311afc970109dee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4362576a57181b2f8b286c839fd7a7a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f3ffec70c057712fb77715a38a4e2c46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9ee5219238ae0bb62f23448218455014(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6e46c5887f35856512b320621eb79ce3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 2, 1024, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_140348328f6cb12bcce84fac7b9197f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e46c5887f35856512b320621eb79ce3
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e4835ee5c4577b10588aabeaf5f8a9ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e46c5887f35856512b320621eb79ce3
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_34597353358d1e51dc0c33f9bf86ab7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 2, 512, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d42ba5852daab8ccd5cf94a6b90cbf02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34597353358d1e51dc0c33f9bf86ab7c
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_93aeea4e7505fcbc8012e1559b924aef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34597353358d1e51dc0c33f9bf86ab7c
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_73f8a7d8da9284291ffc8e44776ecfa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 97, 97], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f0afbf44686d598e536e887674d5541c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 97, 97], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_179f8d007225239922a5b67e4fa6ee28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_58f03670353b5f7179e6132332091f50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a8317dc6dcc81cb91058d1029068d738(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_eb13f6ae57d8e3b8b5ca78b703b1f815(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 86, 3, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_836cc57165ebaff8fea3afdca014fb82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb13f6ae57d8e3b8b5ca78b703b1f815
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a9fb7fdb8691d766a24868fa4ccbdc5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb13f6ae57d8e3b8b5ca78b703b1f815
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fb26dbf988a75a7f164ec6910bb9782b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb13f6ae57d8e3b8b5ca78b703b1f815
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0743a6f53398f37cbe5a9b7e91dcfe6d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 1, 512, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca5d8a32fe3d8a5ff525369c721716f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0743a6f53398f37cbe5a9b7e91dcfe6d
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a900969e9f4ebbbb3147ae6755562269(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0743a6f53398f37cbe5a9b7e91dcfe6d
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cdee69ab42b9d16e1d0f94dcb1688c84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_14d53f860f9464d92eca41c6877197e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0ce44129f0b5564a009657f1641918c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e4e5ada49ab5ada7a2ac34e7b3f8e84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2a763783211d27567c8e641204e3cad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 4, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f818d5a2150be5eee4987aa2357e4a5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 4, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dda2ab120bde3e220bf26c1d07406815(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 4, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15a8e74afae0aa29c79cf73760e7363c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f91f306a5b16821fc31d864318d432b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7d476e0bb9c6517bf217eacc5e59370e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3984d7728c66d6e5b9189fe27ef36f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3024, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8e50b8c60801f909144c2930073cb697(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3024, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e696d40109b35055d6dcd638a444df0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3024, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_002dadd7482422713631be80d814bd15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6e87a2e6e71c5668410cfd1a4d2e8b83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ece5d406700090fc4efd23b6b00d6d47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7f778dc21133fbeebf05bcd06ead4850(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_689d88bdb2b13e85cab68f218adfdf1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a1bcac42002faff9414abac01b1a4fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fd07b86fd417643f138439447fa7e5ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_00a006ec1c39b323a6c9de44d9f28812(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d5d0f9e57d383b74aa1b1cd0c0fa5c21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_28b4383d18995f52f15264369244c884(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b0a32a04a24f91c21a3bdd2b8d694e0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7d6da375dcc5b3bdbfb54b00c97e310f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4e27198636181270466bcf3c2350a358(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d0b2102a5ac937b8215f6eb2e1a68fe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_83a31da4598e6381e9b008142993ad29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_787ddd8d9d49fa50c1fcadb8e48e3d36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1bcddc8cc5ba6648bb963a40becab901(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_716ad5004d706d3eea276b11f665a94c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d7b49ebd4d1c4bcdd3b548805f0e42d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_57c122dee5011b082c1a2153b9ea938f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8042a26f5d0261c1c7a33218ef6875ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e662c7b3d229ffb981829148113d8164(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a593c28c1a8f2f0cf54f64ce450b9e27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cbb901a58aa19815beaae5f01b940e61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7df164e80fde39ec14a76f10b18d4924(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6a2cbffb716ac240fce6b12cd1729a6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a9430a52e19fec29987b54bea23297b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2ff0962f6d2b67ccddf5cd4ca41016f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eb3f90ed48810c3a7935987a56a5b429(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1b600b3a96006c6c4ac6d9fdc9fb871e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_10022b0d0e7ba0498ae2b5a0a68c4b6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_24d94b7d80586b38d10164f1b7b57f40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8533897ab3c52119dd73434bb293aaa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1307fe4f75bb8926228cddb25d5ac51d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_724b0f0239f53490b9acfb8fe4fcbd98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_766007be1334bdad226f728f57c15ed6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_14ae0335fe27c0c658e82eab0df7797c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_38c826658307ecd893cb2034cece4d20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_07635d5cb4ba080b34960306753d8d87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e54952fdfe063a7cb524239c8f294f0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4fd4a3f95d8ae26e157fb2de4b9151c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fb9c2cc04bb68956448569595f059845(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3299a535113d927f5dddebe5d0767698(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e693806105aeaf311b0fc60ed452f5e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c47bec4a7b43dc92a13a5c39eb9046c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9cec71cfda4c979a2ce77634c59fc847(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_09c1bb6c3b76e0b0149159ac6898796b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cd82388534fd0fb3e6c9b564bb07118a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_896f8e0b3263228748d7e644ce7cfab6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84effe0c56b5af7c403be60776c69f55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ea1b09625c6e2e9e95e453bb14328b13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
            paddle.to_tensor([50], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f51c048d3c26ab08ca08cbf7d6a3a40f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([50], dtype='int64').reshape([1]),
            paddle.to_tensor([51], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e0579d5045162083720d155f9054ffaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([51], dtype='int64').reshape([1]),
            paddle.to_tensor([52], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9499e976414c1103accf6f74ae8e5bd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([52], dtype='int64').reshape([1]),
            paddle.to_tensor([53], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7d051b5a7033bbae4f765512e6d7b603(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([53], dtype='int64').reshape([1]),
            paddle.to_tensor([54], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b4cd22af64095421decdd06c71008e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([54], dtype='int64').reshape([1]),
            paddle.to_tensor([55], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4d627a7c36ebb8f8e916c8e734c20176(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([55], dtype='int64').reshape([1]),
            paddle.to_tensor([56], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_44676a46767562d226901eccdccc2a3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([56], dtype='int64').reshape([1]),
            paddle.to_tensor([57], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bd0aa274a344303c14b00dea50ab03df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([57], dtype='int64').reshape([1]),
            paddle.to_tensor([58], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_385a29187a3b792d8402644c398b36ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([58], dtype='int64').reshape([1]),
            paddle.to_tensor([59], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ab796961f0a9fe8deade6d00b5311260(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([59], dtype='int64').reshape([1]),
            paddle.to_tensor([60], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_10568196ab8622006f8f5290a9b24ee6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([60], dtype='int64').reshape([1]),
            paddle.to_tensor([61], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_13e5d9a7e7d57c7223a903896fa2c1bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([61], dtype='int64').reshape([1]),
            paddle.to_tensor([62], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_40b3fc3277aa6c6e48b36a52beda9928(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([62], dtype='int64').reshape([1]),
            paddle.to_tensor([63], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6528b195e0730fdd4473eb2b1b14c804(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([63], dtype='int64').reshape([1]),
            paddle.to_tensor([64], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_986e31123d4b935c00c36cfe46899fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([64], dtype='int64').reshape([1]),
            paddle.to_tensor([65], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1916a9f9c9f7f86793b0553024307a46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([65], dtype='int64').reshape([1]),
            paddle.to_tensor([66], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_23c69331a703265c53ffc237189d8413(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([66], dtype='int64').reshape([1]),
            paddle.to_tensor([67], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d058ea53487cb29678cabbf2f2cccf73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([67], dtype='int64').reshape([1]),
            paddle.to_tensor([68], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_393906d7b5a47812913cb1f3e6f92c6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([68], dtype='int64').reshape([1]),
            paddle.to_tensor([69], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_483f27dfe4ac4904354be535b5b20f4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([69], dtype='int64').reshape([1]),
            paddle.to_tensor([70], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ffc895799ed1d0d8e2cfcdf2bb842e55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([70], dtype='int64').reshape([1]),
            paddle.to_tensor([71], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_37ebbc1d2bd1f8d018246eb0cdd4198f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([71], dtype='int64').reshape([1]),
            paddle.to_tensor([72], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8493beb8c58d928f065d59d355a72ced(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([72], dtype='int64').reshape([1]),
            paddle.to_tensor([73], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_234953b24e08c3a769aaa078e3bebf7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([73], dtype='int64').reshape([1]),
            paddle.to_tensor([74], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4bceef9cd0783f3f725025d64bed2ca7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([74], dtype='int64').reshape([1]),
            paddle.to_tensor([75], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1a3afaeea01a6f2f37aeb094168d1ad7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([75], dtype='int64').reshape([1]),
            paddle.to_tensor([76], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f9fdf3b7b68c76f858fb27ba0369c92a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([76], dtype='int64').reshape([1]),
            paddle.to_tensor([77], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_135452ef1f1bbe5ec6cd9425c12fca0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([77], dtype='int64').reshape([1]),
            paddle.to_tensor([78], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_de978e30106eaa6f7a18a0932946c846(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([78], dtype='int64').reshape([1]),
            paddle.to_tensor([79], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2cea2b6fa757815e63de5df0700a616a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([79], dtype='int64').reshape([1]),
            paddle.to_tensor([80], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_de0d3a63f299f0f08247f338a7b9e4d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([80], dtype='int64').reshape([1]),
            paddle.to_tensor([81], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5b15a45fe4137bb66ada14f247965519(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([81], dtype='int64').reshape([1]),
            paddle.to_tensor([82], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8959563ca6bf530efd735ac8250234fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([82], dtype='int64').reshape([1]),
            paddle.to_tensor([83], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c3aba9bb5cbeaa972774435d7246fbab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([83], dtype='int64').reshape([1]),
            paddle.to_tensor([84], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aa8e66ad707467a9f083b15c31acf229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([84], dtype='int64').reshape([1]),
            paddle.to_tensor([85], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0f2d31daceffed61204b1c983f607740(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([85], dtype='int64').reshape([1]),
            paddle.to_tensor([86], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e2e58a67d372e431d6f986eef0cb41e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([86], dtype='int64').reshape([1]),
            paddle.to_tensor([87], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3b06b4fa098a5af49196a07c0b7b3b77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([87], dtype='int64').reshape([1]),
            paddle.to_tensor([88], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_24b3e753730bd51ef1726a78dd4ff405(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([88], dtype='int64').reshape([1]),
            paddle.to_tensor([89], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1e93bc8f89fce4cb50f1d095e06fdc9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([89], dtype='int64').reshape([1]),
            paddle.to_tensor([90], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6b5327afd5d435c17dfef36a016c0cb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([90], dtype='int64').reshape([1]),
            paddle.to_tensor([91], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_28254893ffa03569a00b91a69ab6e42d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([91], dtype='int64').reshape([1]),
            paddle.to_tensor([92], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d858f26db49bf56d2ca35a54b7a403da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([92], dtype='int64').reshape([1]),
            paddle.to_tensor([93], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b55f772d1de9b481cdc437404a8048f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([93], dtype='int64').reshape([1]),
            paddle.to_tensor([94], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cdd36c6c3ab79dae39dbd96421f27c20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([94], dtype='int64').reshape([1]),
            paddle.to_tensor([95], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6c2ff9a5a5d125e9468f4d2250a98d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([95], dtype='int64').reshape([1]),
            paddle.to_tensor([96], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c1eebbdbf109b410418d3b99e201d1a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([96], dtype='int64').reshape([1]),
            paddle.to_tensor([97], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_45c4160ceb86592df7afdd35a9216228(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([97], dtype='int64').reshape([1]),
            paddle.to_tensor([98], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2943342b16ab63f5a757901e8392e2ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([98], dtype='int64').reshape([1]),
            paddle.to_tensor([99], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_83fd5d1a348ecf0cf91b788546458ef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([99], dtype='int64').reshape([1]),
            paddle.to_tensor([100], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_da4aaad4d8b0f6c0fe76358f8b930257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([100], dtype='int64').reshape([1]),
            paddle.to_tensor([101], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cad62498755bd72c8f346bff57b15208(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([101], dtype='int64').reshape([1]),
            paddle.to_tensor([102], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_14e1f8167f0b54bb648f45820edae159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([102], dtype='int64').reshape([1]),
            paddle.to_tensor([103], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9b371459bc9945567c4f10e5a26a8a58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([103], dtype='int64').reshape([1]),
            paddle.to_tensor([104], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d981e6461beefa6ef4a6653d75c476c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([104], dtype='int64').reshape([1]),
            paddle.to_tensor([105], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8fea947aec8c85b491e79942bf084695(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([105], dtype='int64').reshape([1]),
            paddle.to_tensor([106], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0ebeede4c550fb7476124a0d72ddf26d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([106], dtype='int64').reshape([1]),
            paddle.to_tensor([107], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e596193053f33f8ea4e514a5b5d42d9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([107], dtype='int64').reshape([1]),
            paddle.to_tensor([108], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_26def687ad6b6203056dc28ceedf8311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([108], dtype='int64').reshape([1]),
            paddle.to_tensor([109], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_737b57cccdace04aa8da2d942939115d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([109], dtype='int64').reshape([1]),
            paddle.to_tensor([110], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7468458ad78e88a21116506c52d6f206(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([110], dtype='int64').reshape([1]),
            paddle.to_tensor([111], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d8ce950f55cef3df45552e7ffff1280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([111], dtype='int64').reshape([1]),
            paddle.to_tensor([112], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5d482a3a530fad376c5d5f216e094587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([112], dtype='int64').reshape([1]),
            paddle.to_tensor([113], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_edf0bcacee4b2c10fe62e8f2c4059b27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([113], dtype='int64').reshape([1]),
            paddle.to_tensor([114], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e425c6fc5bf5251a4433717aaf6e22c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([114], dtype='int64').reshape([1]),
            paddle.to_tensor([115], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f23964f28cefd78e07df5ef7733f7f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([115], dtype='int64').reshape([1]),
            paddle.to_tensor([116], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b7f5706a448de537cb1b90897365b5cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([116], dtype='int64').reshape([1]),
            paddle.to_tensor([117], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bb2a5d1f515d059ee5fc9777232725cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([117], dtype='int64').reshape([1]),
            paddle.to_tensor([118], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_156697a3fcd07b32b42c1d458ef3170d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([118], dtype='int64').reshape([1]),
            paddle.to_tensor([119], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c79e2993cfc5f6613d0cefb1c7ea26d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([119], dtype='int64').reshape([1]),
            paddle.to_tensor([120], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d43fe48da271b3c28459abb2525d0540(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([120], dtype='int64').reshape([1]),
            paddle.to_tensor([121], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1e4fad5900e2d20e651229566cee5cef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([121], dtype='int64').reshape([1]),
            paddle.to_tensor([122], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_df94a414d1e4662187ad2b46561ed914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([122], dtype='int64').reshape([1]),
            paddle.to_tensor([123], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a1f1c7cca9aec3ebe5dee55ad87ba864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([123], dtype='int64').reshape([1]),
            paddle.to_tensor([124], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2f8ac6706feaf29fc473f890bd0ff3fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([124], dtype='int64').reshape([1]),
            paddle.to_tensor([125], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f4a8b5cd45f32a3bcb385d0c7083cf4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([125], dtype='int64').reshape([1]),
            paddle.to_tensor([126], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0c0fdbf41feba358253916f340ad76a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([126], dtype='int64').reshape([1]),
            paddle.to_tensor([127], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_43507d0bfb618e0ba0467e2f8d18307c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([127], dtype='int64').reshape([1]),
            paddle.to_tensor([128], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_05f9d1648f3c778e8a8e2dd89744c9b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([128], dtype='int64').reshape([1]),
            paddle.to_tensor([129], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_20ec54a017ca61289739c36d0a0e05af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([129], dtype='int64').reshape([1]),
            paddle.to_tensor([130], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9494d24fc79e97b71340756de4636bda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([130], dtype='int64').reshape([1]),
            paddle.to_tensor([131], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bcfe2b9b87a3224ea0ddcbd24e0c5091(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([131], dtype='int64').reshape([1]),
            paddle.to_tensor([132], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4f0f02de8fc112b433f3e6b1b7811752(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([132], dtype='int64').reshape([1]),
            paddle.to_tensor([133], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4f1af25e31baecd3088075a19b1c3c10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([133], dtype='int64').reshape([1]),
            paddle.to_tensor([134], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8e0d21faee8b2961b03532f813bb5626(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([134], dtype='int64').reshape([1]),
            paddle.to_tensor([135], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1586eec8f5993c61219486e0c627f5c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([135], dtype='int64').reshape([1]),
            paddle.to_tensor([136], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_db7d56dcd82233ef55a8e9b6039c6663(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([136], dtype='int64').reshape([1]),
            paddle.to_tensor([137], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e6fa05bb4b912e888de89be36a1ae07e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([137], dtype='int64').reshape([1]),
            paddle.to_tensor([138], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_91ad0768c027d9cb6a4340d745cdacc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([138], dtype='int64').reshape([1]),
            paddle.to_tensor([139], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4742d4649894cacdece25f0c1cc265dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([139], dtype='int64').reshape([1]),
            paddle.to_tensor([140], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_46f855a38d3f4ed8e349cffa10b88bd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([140], dtype='int64').reshape([1]),
            paddle.to_tensor([141], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_55fd51821cf6d12703feb30c4759cae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([141], dtype='int64').reshape([1]),
            paddle.to_tensor([142], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c3086a65a83f3b8dcd716da4b8a3d87b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([142], dtype='int64').reshape([1]),
            paddle.to_tensor([143], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cd12d0ce2758111e721fe6888ced69a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([143], dtype='int64').reshape([1]),
            paddle.to_tensor([144], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8c1b1e8d81286f0fcf481b96b3716a4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([144], dtype='int64').reshape([1]),
            paddle.to_tensor([145], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_621e8fed87d9cf203167886ed7e5c258(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([145], dtype='int64').reshape([1]),
            paddle.to_tensor([146], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1d0fe88b72a295be6b58d313e7cad48e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([146], dtype='int64').reshape([1]),
            paddle.to_tensor([147], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7bc7a4eeed8d1b9007d91896cf2cea8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([147], dtype='int64').reshape([1]),
            paddle.to_tensor([148], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a5b8ff92b3a5c6f6fdadd9203f5e33c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([148], dtype='int64').reshape([1]),
            paddle.to_tensor([149], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_51d0d17cf63063b0dc4579370fad8d17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([149], dtype='int64').reshape([1]),
            paddle.to_tensor([150], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9de55dc416e8077d133b0393a775a341(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([150], dtype='int64').reshape([1]),
            paddle.to_tensor([151], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_db59d47dc5864e936ed82ebbe2076798(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([151], dtype='int64').reshape([1]),
            paddle.to_tensor([152], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_98e7188aa3cb9b079d9d81da5caa47f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([152], dtype='int64').reshape([1]),
            paddle.to_tensor([153], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b42f4aeb34fba66e833a97b215503345(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([153], dtype='int64').reshape([1]),
            paddle.to_tensor([154], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8757f88f95ce455f90a09304a95c5ad9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([154], dtype='int64').reshape([1]),
            paddle.to_tensor([155], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f41a15f98879c4208541a0d511afb623(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([155], dtype='int64').reshape([1]),
            paddle.to_tensor([156], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cb79e299ae7e2a4e4f0cbb40bc5b989e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([156], dtype='int64').reshape([1]),
            paddle.to_tensor([157], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7e543f5c7767058710fb5d73399b185a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([157], dtype='int64').reshape([1]),
            paddle.to_tensor([158], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f2995f374a91805f59641a59d76fbae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([158], dtype='int64').reshape([1]),
            paddle.to_tensor([159], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_527d40303f403af3ee85f50a3d3237aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([159], dtype='int64').reshape([1]),
            paddle.to_tensor([160], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8408adc6d72c491ef86753faba813533(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([160], dtype='int64').reshape([1]),
            paddle.to_tensor([161], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_516e9293cc9cf29067a0b97539e02e13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([161], dtype='int64').reshape([1]),
            paddle.to_tensor([162], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_74e3f3f3e2e01a52690afaebf7e4dbbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([162], dtype='int64').reshape([1]),
            paddle.to_tensor([163], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0176e617831d58e0783f39d928684e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([163], dtype='int64').reshape([1]),
            paddle.to_tensor([164], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fa498b83b3905fa38aec5cfbcd37252b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([164], dtype='int64').reshape([1]),
            paddle.to_tensor([165], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1893a88c4595a11a680b4971960f5f6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([165], dtype='int64').reshape([1]),
            paddle.to_tensor([166], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ed10e855ecca2931d37ff806b3f18e16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([166], dtype='int64').reshape([1]),
            paddle.to_tensor([167], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_89e6424020f23bb4115aeb5ab8d93a86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([167], dtype='int64').reshape([1]),
            paddle.to_tensor([168], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_11f6b39d82415f61d3bf475168b05817(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([168], dtype='int64').reshape([1]),
            paddle.to_tensor([169], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_473233ce124b4299262dae19d370186e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([169], dtype='int64').reshape([1]),
            paddle.to_tensor([170], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cc40c7297f01c57c3d9992fc26712b73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([170], dtype='int64').reshape([1]),
            paddle.to_tensor([171], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_85f5a0553ef2e17991b455a52ea09c41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([171], dtype='int64').reshape([1]),
            paddle.to_tensor([172], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86144bd9af9697c0a6412bbec213dfa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([172], dtype='int64').reshape([1]),
            paddle.to_tensor([173], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d9b710973f36db08e2ab0f3e1fa31e38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([173], dtype='int64').reshape([1]),
            paddle.to_tensor([174], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7e66f5a658dd4fbf56f05a469e272f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([174], dtype='int64').reshape([1]),
            paddle.to_tensor([175], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1e8bab3144d270dd4de0b4b3933f21ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([175], dtype='int64').reshape([1]),
            paddle.to_tensor([176], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5dd1182d4699e739b1e143083349be4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([176], dtype='int64').reshape([1]),
            paddle.to_tensor([177], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5650fa3b1211ff62e5202407fc4f1d9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([177], dtype='int64').reshape([1]),
            paddle.to_tensor([178], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bd070239a614da61a2145e77b1fc1b59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([178], dtype='int64').reshape([1]),
            paddle.to_tensor([179], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2b2b1cb9b515121e3ad06a804f4cf517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([179], dtype='int64').reshape([1]),
            paddle.to_tensor([180], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4d7fcc68af6345f0d8f12302a4bdc5f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([180], dtype='int64').reshape([1]),
            paddle.to_tensor([181], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b614d79f97af223c166c83f2b77b4418(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([181], dtype='int64').reshape([1]),
            paddle.to_tensor([182], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c1daa9b2f829691272f582813df98c68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([182], dtype='int64').reshape([1]),
            paddle.to_tensor([183], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d031ff4f2745aeaef5fdfa3034503a51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([183], dtype='int64').reshape([1]),
            paddle.to_tensor([184], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1c5b797be4231cc91d3c52c97b388738(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([184], dtype='int64').reshape([1]),
            paddle.to_tensor([185], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf0d48b616fd86c1d79b00a8a11f787b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([185], dtype='int64').reshape([1]),
            paddle.to_tensor([186], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a909972a029da4815ccebd00f1d5b8bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([186], dtype='int64').reshape([1]),
            paddle.to_tensor([187], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a49443bb7de3fd75ad944df5f6aabdf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([187], dtype='int64').reshape([1]),
            paddle.to_tensor([188], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_caea9abfdbb504e84f31a4a02c72f728(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([188], dtype='int64').reshape([1]),
            paddle.to_tensor([189], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6d6802ae201e02b930c82f504496d6f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([189], dtype='int64').reshape([1]),
            paddle.to_tensor([190], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4665b193860543dab6c08b2d13756064(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([190], dtype='int64').reshape([1]),
            paddle.to_tensor([191], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8a8b98fa17462debabe43811e9fa0369(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([191], dtype='int64').reshape([1]),
            paddle.to_tensor([192], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_019e0109bb854454302bc1c214db64fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([192], dtype='int64').reshape([1]),
            paddle.to_tensor([193], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7fb8d3a8bc03531d93ecc174309128f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([193], dtype='int64').reshape([1]),
            paddle.to_tensor([194], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ec67aeb59aa32399d6953e74cbcf5ac0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([194], dtype='int64').reshape([1]),
            paddle.to_tensor([195], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ee33cfda4352a467e7262f29782ddb51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([195], dtype='int64').reshape([1]),
            paddle.to_tensor([196], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_25c9b174fa4874bf8e39115e8d94ebeb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 10, 8, 160, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c1838feaf6069b1abace6781cbacc578(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25c9b174fa4874bf8e39115e8d94ebeb
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4fa43fc2bd212b1185d75197076255cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25c9b174fa4874bf8e39115e8d94ebeb
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_28416776a0773c9b9036ca778b8428db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25c9b174fa4874bf8e39115e8d94ebeb
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_98b60d53a8a026bf86dc3ee51baf9a34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f3219fd8f9255a387d5732ecfb5dd3bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bd0659c3bb7dc5f4ab8114a69523c6c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d8b22300601cf0797ba0716d057796d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_55b910ae92e57cdc29dc20445cb5351b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5821260db38fe66e09e48663a9f627c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_df118dc1f5939ee2129d232d40b65a53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_685a4e3444ca19ce3c54e4938dfbca94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4110d132f324102d93fee9d8b4f7b500(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_976fa3a9679b669722b621e07bb12c34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3551f3a6a1fbb09865b36cf2d33964a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0bbc17267468bce3a0bcdc29e387e105(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c427364a09e501ead93fd18bfe99124c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5650c05909174bbf1408e20870923f09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d63b5c3a2c7b8fb9b3adc9c0c2eda2ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_df86c8c4d2a293c7646fcca0a7f08b20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_99a696b901413375725fde611ebe0938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fcb6ea1cb9647da7012cd89b928c9ef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_39da1bc3e34e2cfdd4cb88b43f88f26f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4353234b7187264a7ea4a030aa463eae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5a7f7b2ca65041d2ba333c3c32f26bed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3a7342ab1b4620f02c1c20ca9c1dc09a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b00b6113e1ebe1f57721fbc04406ebf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8851de3d1cd4194f36f53c807d3deebc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_998e8db0b2c6ff692a512681dfa4c777(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a319c3594b73f190439b81adc4b7c347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9ff57fe4a3ce4174a851d4f51f9f5269(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_837567efaa39c8502cb059214af33f25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_40b205508c4ed0c9f52f1b1cb381b095(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c0b5b98324814afc89aaef63bdbe8501(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b170d3975e1ac98904a31f22cae6d962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ecda9c5d3f709e37e49ff136cabfac9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2ca8194ab3c65d0f286cd37a1a1f2257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a46f1791d93d68d410dc6950b1c0be9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_32ea7295298a13d78dddbd883686c4f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c7249e799fb0c854b8741dce09873bab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7df2ae77afa489e694c9775064085f26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0aebdccbb9f2d0531b8ddac1b015d0b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_33eac82067387ab7775913763552e011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7e3f0b3d1b60d28e867001da33c32734(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a1f9344133f31748dd340bccb6efa944(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_031e6b2ce71245105abf8437717026d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_383772468bd672ad992ea6bc50c32a4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4d9e58f108cd5fc99719be2aa22c254b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b4e1cfa75c897bd655a86ed5d80ddd91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f037b2da240c04f62afe2f8af3a2e179(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_19a19221cc4c7ac99d2cf9b1afa64347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fad7963f43974f50f1ec68a4227a4933(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b137f2580bd81b646d38e392c033d55b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 16], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf6ee31a6d877189a45046d939ce1142(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a8d705a8b8e830a08f7bce09b9794df8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2ee91842a0250225b282eebcb4a77979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_389f553a78453cd73c325702d210c7fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ce792f1b9af7e159cc5d593a63547291(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_711a7ca83d2b3581f78d5e08a791393f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3359d2562bb7845572eb32b5a775ffa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_13ae7414bd67524ee177346a37e699ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_523ad7c6ccd679f169f54cb0d15c8d73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8ced09221f5d308e0ad6a6865284812c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ed22aa6b2e76f030e66a0cb620d655c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d4a557579ce909af6cbd8495f5dca869(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_be0dea0ff20b4caec982773aecfe4e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ec55008ed0616ec381a8a3470729cd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_97f834b8336bdd39f2720cfa95fc3855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_82d597f311c2836675debc5103302268(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f6b8eda40c93410cca910de04559210d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1174, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d24a52f1fc30700b34046f0ff632810e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1174, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ca1e073ed5d4874d2a0c735d9fe97943(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 1, 6, 1174, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1c1653a3c2224c837794d64ebd4a9ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca1e073ed5d4874d2a0c735d9fe97943
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3ddfddd85859bf1a4aafb2625c43c556(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca1e073ed5d4874d2a0c735d9fe97943
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_519c681a08e102b5725811cc4f2afff5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca1e073ed5d4874d2a0c735d9fe97943
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cdee69ab42b9d16e1d0f94dcb1688c84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_14d53f860f9464d92eca41c6877197e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8c2e5dd97a0a07e55847057840fd7acc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fe14618241ff4c15c3d14907708e5b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bcdd7b3c4bf858ce88ec2d32f5f9d87b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_859614572eb1d4eeaa9f578d68ed3c1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ea2b075f277df0f9769c469036c0260f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d13ae9b0bd9a1f97a7321da9f1eae7d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 116, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5d31fcadde73adcef820d62b138b1b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 116, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9497f4c834cf08e1c061b36fc69488c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 116, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_06975b6c219b338076e2ecf5cdca495c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 116, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f64b4ac90a112a72ab5aef8d3f84e000(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1174, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_52d163218a6c5c0bdc8f0cce85f58262(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1174, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9c2847df77f89a6a8960598823b2b88e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 1, 12, 1174, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d96b9a783446710241a9ab236aefc9d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c2847df77f89a6a8960598823b2b88e
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c40a9c5f30424b0a96a36407131ba1a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c2847df77f89a6a8960598823b2b88e
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2c6c048c67ffe1acd7b36d9f9670769a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c2847df77f89a6a8960598823b2b88e
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1d0715b9a9013430bae6075ed89327fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 1, 1024, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5b533ac45ebfe99198d78e44b349346a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d0715b9a9013430bae6075ed89327fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4e8bf6097b5bff8ac5fb640ce61e8571(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d0715b9a9013430bae6075ed89327fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_419ee6cd420a6bb006e85d23a1db0f7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_093570a2bb1492dec51b2c5ba71cc991(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e8c81928ea99514bf24cfe838ab4402a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a5e8acfcb911a4f2ae64f8e695651a5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_72b2e2c83c6579c8f538232ce3427874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bdf87d212c9f03e1c6115609d9e456de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a565e967cfc3e2656038e87e470ebab5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eed6280fe4d41f4549cbe667f60124dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8b7fa2c06ded95d7d026041de9a87934(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4bd17a08b01a11f2269f9761bd9e2ba7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_addef1c625a81e391746a8da4f67afcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ad4e5a4be52446e086f09ffe7e8835a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ca86975b97ab3e76fae5640fc761a45c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_812d4312316eca05de33a19563d7f7c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6e746a36c0301f9a4a7e8166c91ee
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5f00a27711b55e2c6c6732fabb722747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_714dca9551891de5c71f29cdf77297ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 10, 8, 50, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c2083486996fe55dee15958013c61188(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714dca9551891de5c71f29cdf77297ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_af5c93f2c31f66180afffa6934103a85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714dca9551891de5c71f29cdf77297ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a91e3f1b8da53923a8364932b3601539(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714dca9551891de5c71f29cdf77297ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()