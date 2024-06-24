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



class PrimitiveOp_8612d0d68b8a650001a86691bd316818(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0828494eb8031b6ee7bef0d63eb29f4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8612d0d68b8a650001a86691bd316818
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0828494eb8031b6ee7bef0d63eb29f4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8612d0d68b8a650001a86691bd316818
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c310cce4553ed7afb0f29ab5043cbdb8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7162c440bd08af492690061cd4d91db7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c310cce4553ed7afb0f29ab5043cbdb8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7162c440bd08af492690061cd4d91db7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c310cce4553ed7afb0f29ab5043cbdb8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3213aced3839e6653017d5749c904e32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8a74c1060a83bb05ec8d2c868e0de9e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3213aced3839e6653017d5749c904e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a74c1060a83bb05ec8d2c868e0de9e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3213aced3839e6653017d5749c904e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cdba59fb7e86bd9a3f073cbfd82e0b47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ec501ce26427fcf4e322b6a7c478956(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdba59fb7e86bd9a3f073cbfd82e0b47
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ec501ce26427fcf4e322b6a7c478956(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdba59fb7e86bd9a3f073cbfd82e0b47
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_92a93e26fdabbeac29ed6b50f4b9a312(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f45386334524942c4d2492b8f5e267f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92a93e26fdabbeac29ed6b50f4b9a312
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f45386334524942c4d2492b8f5e267f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92a93e26fdabbeac29ed6b50f4b9a312
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5ef48365d12968559582b82ede7e159a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_514e47f363cb7b8408b7fb4c18855bd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef48365d12968559582b82ede7e159a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_514e47f363cb7b8408b7fb4c18855bd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef48365d12968559582b82ede7e159a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0828494eb8031b6ee7bef0d63eb29f4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8612d0d68b8a650001a86691bd316818
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0828494eb8031b6ee7bef0d63eb29f4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8612d0d68b8a650001a86691bd316818
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2850aeb421d78f007bb2e5ce26a7c30c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_84d5eb269bfd34f9bc32ecb9f4891a02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2850aeb421d78f007bb2e5ce26a7c30c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_84d5eb269bfd34f9bc32ecb9f4891a02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2850aeb421d78f007bb2e5ce26a7c30c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_879f6a4319440687426b9e356ad5e013(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_918bc743367792c6d670100994688937(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_879f6a4319440687426b9e356ad5e013
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_918bc743367792c6d670100994688937(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_879f6a4319440687426b9e356ad5e013
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_50f230518a0809bc0100e06e8ae80d8b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1786, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1786, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6a4bb4752746fe50cb8c37f8887987e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50f230518a0809bc0100e06e8ae80d8b
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a4bb4752746fe50cb8c37f8887987e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50f230518a0809bc0100e06e8ae80d8b
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a4bb4752746fe50cb8c37f8887987e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50f230518a0809bc0100e06e8ae80d8b
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a4bb4752746fe50cb8c37f8887987e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50f230518a0809bc0100e06e8ae80d8b
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9338121665dabf5961b040759d84cfa3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_73bf95fa1bb4809922811d509d0a4cde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9338121665dabf5961b040759d84cfa3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73bf95fa1bb4809922811d509d0a4cde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9338121665dabf5961b040759d84cfa3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b629aeb02983c44eb0a04a805e9a408d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_603ce0506d4f1da60473de5f42c47f09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b629aeb02983c44eb0a04a805e9a408d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_603ce0506d4f1da60473de5f42c47f09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b629aeb02983c44eb0a04a805e9a408d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1c8282bfc8d9625c71eb37c7566f5842(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_932fe2d81e8dac5fc9080630b2d3ae8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c8282bfc8d9625c71eb37c7566f5842
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_932fe2d81e8dac5fc9080630b2d3ae8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c8282bfc8d9625c71eb37c7566f5842
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b812e843f25f9f629b3ed8ee38ba590d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_234d8c13e4e8a4ee2586eedb89af8b47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b812e843f25f9f629b3ed8ee38ba590d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_234d8c13e4e8a4ee2586eedb89af8b47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b812e843f25f9f629b3ed8ee38ba590d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f2f5d16d66ff37d07e245841e29a95df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0944250226020813], [0.43199148774147034], [0.30663737654685974], [0.30789077281951904], [0.017560848966240883], [0.4452926218509674], [0.3800290524959564], [0.212552011013031], [0.06295822560787201]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.4870230257511139], [0.41575801372528076], [0.09537261724472046], [0.13353395462036133], [0.23935119807720184], [0.10870978981256485], [0.15444859862327576], [0.14588458836078644], [0.23597005009651184]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_fa14dd9722d82f23c8b8c87c2cba34bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.46759042143821716], [0.05409188196063042], [0.31611233949661255], [0.4931334853172302], [0.3475216031074524], [0.33039727807044983], [0.12327843904495239], [0.307874858379364], [0.3336060643196106]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.15379993617534637], [0.34884482622146606], [0.3303232789039612], [0.3350580632686615], [0.24807313084602356], [0.003155889455229044], [0.48330458998680115], [0.15179887413978577], [0.044809091836214066]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_5abe4a2a97ba22158f81b09fe537177f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2622634470462799], [0.2463095784187317], [0.40173643827438354], [0.3592231869697571], [0.3121943771839142], [0.14940416812896729], [0.09890615940093994], [0.3055746257305145], [0.06899479031562805]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.08312228322029114], [0.009725884534418583], [0.31125256419181824], [0.18358245491981506], [0.14136718213558197], [0.22684775292873383], [0.2601875066757202], [0.1880447119474411], [0.2560676038265228]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_a0f586446315ffe0f277a19ba99876db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3488651216030121], [0.4772522747516632], [0.07919348776340485], [0.36751043796539307], [0.10987333208322525], [0.11728937923908234], [0.13010436296463013], [0.21030104160308838], [0.2969972491264343]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.07335024327039719], [0.072712242603302], [0.08202075958251953], [0.35215020179748535], [0.43682003021240234], [0.356018990278244], [0.25737234950065613], [0.4559156596660614], [0.34402239322662354]], dtype='float32').reshape([9, 1]),
        ]


class PrimitiveOp_9b0e4723ee208039b0e4b391026094bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5529, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5529, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4158c8cc2ebc2171a5de46b344d8a6ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b0e4723ee208039b0e4b391026094bf
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4158c8cc2ebc2171a5de46b344d8a6ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b0e4723ee208039b0e4b391026094bf
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4158c8cc2ebc2171a5de46b344d8a6ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b0e4723ee208039b0e4b391026094bf
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4158c8cc2ebc2171a5de46b344d8a6ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b0e4723ee208039b0e4b391026094bf
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3b6f51e0ff307ce9e9ae0a6d3c1cc52e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            paddle.static.InputSpec(shape=[6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_add367617ddcfab3e7986f94727edcb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6f51e0ff307ce9e9ae0a6d3c1cc52e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4773276150226593, 0.4900837242603302, 0.31976962089538574, 0.23725539445877075, 0.3461141586303711, 0.47337961196899414], dtype='float32').reshape([6]),
            paddle.to_tensor([0.09207890182733536, 0.23877081274986267, 0.14726030826568604, 0.3021329939365387, 0.4564148485660553, 0.3768220543861389], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_13bd28e28adba21cc73ff3a23e58e9f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6f51e0ff307ce9e9ae0a6d3c1cc52e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12860596179962158, 0.45978280901908875, 0.49666598439216614, 0.21115154027938843, 0.47597774863243103, 0.44470906257629395], dtype='float32').reshape([6]),
            paddle.to_tensor([0.32528331875801086, 0.010651743039488792, 0.30837002396583557, 0.4661112129688263, 0.16577965021133423, 0.012630387209355831], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_e772c8d318325a6277264ac6a42f03f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6f51e0ff307ce9e9ae0a6d3c1cc52e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09849236905574799, 0.377002090215683, 0.03728121519088745, 0.03544168174266815, 0.3461141586303711, 0.37609827518463135], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2667119801044464, 0.46009713411331177, 0.48630520701408386, 0.3197534382343292, 0.08830360323190689, 0.18910035490989685], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_057129da447cdfe639143fd27a15d340(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6f51e0ff307ce9e9ae0a6d3c1cc52e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10391133278608322, 0.45978280901908875, 0.49666598439216614, 0.21115154027938843, 0.31940513849258423, 0.33928123116493225], dtype='float32').reshape([6]),
            paddle.to_tensor([0.33211463689804077, 0.06320647895336151, 0.03932555019855499, 0.3673308491706848, 0.05595792829990387, 0.18198029696941376], dtype='float32').reshape([6]),
        ]


class PrimitiveOp_da77ac58e61d45fefe02323f8803a1a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1767, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1767, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c601176c0d6001739949eb9dacbb8fd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da77ac58e61d45fefe02323f8803a1a6
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c601176c0d6001739949eb9dacbb8fd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da77ac58e61d45fefe02323f8803a1a6
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c601176c0d6001739949eb9dacbb8fd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da77ac58e61d45fefe02323f8803a1a6
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c601176c0d6001739949eb9dacbb8fd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da77ac58e61d45fefe02323f8803a1a6
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ec501ce26427fcf4e322b6a7c478956(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdba59fb7e86bd9a3f073cbfd82e0b47
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ec501ce26427fcf4e322b6a7c478956(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdba59fb7e86bd9a3f073cbfd82e0b47
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_603ce0506d4f1da60473de5f42c47f09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b629aeb02983c44eb0a04a805e9a408d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_603ce0506d4f1da60473de5f42c47f09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b629aeb02983c44eb0a04a805e9a408d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f65e740fdbed466481ae6da55f877a3f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1490, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1490, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_44e55e3839362524b26e20cddcaf034d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f65e740fdbed466481ae6da55f877a3f
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44e55e3839362524b26e20cddcaf034d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f65e740fdbed466481ae6da55f877a3f
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44e55e3839362524b26e20cddcaf034d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f65e740fdbed466481ae6da55f877a3f
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44e55e3839362524b26e20cddcaf034d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f65e740fdbed466481ae6da55f877a3f
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_514e47f363cb7b8408b7fb4c18855bd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef48365d12968559582b82ede7e159a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_514e47f363cb7b8408b7fb4c18855bd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef48365d12968559582b82ede7e159a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7162c440bd08af492690061cd4d91db7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c310cce4553ed7afb0f29ab5043cbdb8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7162c440bd08af492690061cd4d91db7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c310cce4553ed7afb0f29ab5043cbdb8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_08a10ef183893e237ec1181bcc09d4f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.41494059562683105]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3083866834640503]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_8a8c7d9c3e9efeeac0506460e3623e0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16263581812381744]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3545854091644287]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_4a55ef22ae9430f8f8ec7be59bdcb991(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10528379678726196]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3151826858520508]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_c43b16052e44fc249eeba965d38431bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07341162115335464]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.21531380712985992]], dtype='float32').reshape([1, 1]),
        ]


class PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cb4155c70c9079d79e6ddb86ea52e68d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.376817524433136], [0.45837101340293884], [0.4508039653301239], [0.24826665222644806], [0.04853551834821701], [0.25887274742126465]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.04222099483013153], [0.4947202205657959], [0.46678364276885986], [0.2570192813873291], [0.16303293406963348], [0.3793601393699646]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_76a9f548c5cc2c1bcea5ad9f17c613aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2849709689617157], [0.4665500521659851], [0.4074956178665161], [0.22639164328575134], [0.21449042856693268], [0.04977961629629135]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3085649907588959], [0.11111163347959518], [0.23699478805065155], [0.22173336148262024], [0.05751029774546623], [0.39112553000450134]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_6046cf7894bf61bea2dfa8cb0eab23b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10129960626363754], [0.42914801836013794], [0.3438938558101654], [0.010731011629104614], [0.2497348189353943], [0.2733650505542755]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.09891922771930695], [0.4956405460834503], [0.13846367597579956], [0.49526292085647583], [0.09620732814073563], [0.3659272789955139]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_20b31f5b4272e5dc1349d9fecc8f8ee4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.31633827090263367], [0.34771016240119934], [0.2228093296289444], [0.22192063927650452], [0.43225693702697754], [0.3838341534137726]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3794229328632355], [0.35555100440979004], [0.0830652266740799], [0.07343913614749908], [0.06483077257871628], [0.41214850544929504]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_73bf95fa1bb4809922811d509d0a4cde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9338121665dabf5961b040759d84cfa3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73bf95fa1bb4809922811d509d0a4cde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9338121665dabf5961b040759d84cfa3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a74c1060a83bb05ec8d2c868e0de9e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3213aced3839e6653017d5749c904e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a74c1060a83bb05ec8d2c868e0de9e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3213aced3839e6653017d5749c904e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1b2ab28214ca21398a9d39ed13a825fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2010, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2010, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_607a108f29cfdfa8382d28b818835c12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b2ab28214ca21398a9d39ed13a825fd
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_607a108f29cfdfa8382d28b818835c12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b2ab28214ca21398a9d39ed13a825fd
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_607a108f29cfdfa8382d28b818835c12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b2ab28214ca21398a9d39ed13a825fd
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_607a108f29cfdfa8382d28b818835c12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b2ab28214ca21398a9d39ed13a825fd
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1df01a60f25e84fe7f62858a629f443f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d3ea942f59bdaf3924bcf57deaa7a0dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df01a60f25e84fe7f62858a629f443f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3ea942f59bdaf3924bcf57deaa7a0dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df01a60f25e84fe7f62858a629f443f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_151d6c9cba5d3a7a5d6d0809bfc799d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4663, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4663, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_435c11de00f82f534c575a6241c7e874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_151d6c9cba5d3a7a5d6d0809bfc799d4
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_435c11de00f82f534c575a6241c7e874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_151d6c9cba5d3a7a5d6d0809bfc799d4
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_435c11de00f82f534c575a6241c7e874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_151d6c9cba5d3a7a5d6d0809bfc799d4
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_435c11de00f82f534c575a6241c7e874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_151d6c9cba5d3a7a5d6d0809bfc799d4
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bfadff41446eee305f0ee9b04ae2d9d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1090, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1090, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ba14802e2c99b386248da47ac4049ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfadff41446eee305f0ee9b04ae2d9d5
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba14802e2c99b386248da47ac4049ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfadff41446eee305f0ee9b04ae2d9d5
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba14802e2c99b386248da47ac4049ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfadff41446eee305f0ee9b04ae2d9d5
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba14802e2c99b386248da47ac4049ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfadff41446eee305f0ee9b04ae2d9d5
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f45386334524942c4d2492b8f5e267f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92a93e26fdabbeac29ed6b50f4b9a312
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f45386334524942c4d2492b8f5e267f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92a93e26fdabbeac29ed6b50f4b9a312
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_932fe2d81e8dac5fc9080630b2d3ae8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c8282bfc8d9625c71eb37c7566f5842
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_932fe2d81e8dac5fc9080630b2d3ae8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c8282bfc8d9625c71eb37c7566f5842
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5ddd4e6b279ada749248efad1383ba3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0807226225733757], [0.26894018054008484], [0.30494368076324463], [0.27619537711143494], [0.428976833820343]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.42150676250457764], [0.40049660205841064], [0.22837966680526733], [0.09048057347536087], [0.3214188814163208]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_2939d7fb5a11507a78fa67e6adc75610(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.23073291778564453], [0.08885031193494797], [0.22880247235298157], [0.4462369680404663], [0.41620945930480957]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.15623098611831665], [0.23012053966522217], [0.0010281642898917198], [0.315184623003006], [0.44070783257484436]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_7c784e83d1daf22f9c0be1d91289e9ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06819140911102295], [0.0015212241560220718], [0.29410940408706665], [0.0312783420085907], [0.05078583583235741]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.35739219188690186], [0.11215049773454666], [0.24850419163703918], [0.08986014872789383], [0.42450934648513794]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_b6e2f76c1096aba5b450fe31f1973114(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.34516990184783936], [0.14900963008403778], [0.23224982619285583], [0.4999895989894867], [0.029579993337392807]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.2559264004230499], [0.4242517352104187], [0.3585534393787384], [0.4039344787597656], [0.33067619800567627]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_918bc743367792c6d670100994688937(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_879f6a4319440687426b9e356ad5e013
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_918bc743367792c6d670100994688937(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_879f6a4319440687426b9e356ad5e013
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_234d8c13e4e8a4ee2586eedb89af8b47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b812e843f25f9f629b3ed8ee38ba590d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_234d8c13e4e8a4ee2586eedb89af8b47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b812e843f25f9f629b3ed8ee38ba590d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a860fac71210374cca5ca09f1371af03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_432df3797e5f57023b82fab6a50fa7e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a860fac71210374cca5ca09f1371af03
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_432df3797e5f57023b82fab6a50fa7e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a860fac71210374cca5ca09f1371af03
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a79ff053f93f4873c4d34cc37cad3deb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2374, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2374, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3af57128f49765e178d4862ce371766f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a79ff053f93f4873c4d34cc37cad3deb
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3af57128f49765e178d4862ce371766f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a79ff053f93f4873c4d34cc37cad3deb
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3af57128f49765e178d4862ce371766f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a79ff053f93f4873c4d34cc37cad3deb
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3af57128f49765e178d4862ce371766f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a79ff053f93f4873c4d34cc37cad3deb
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b66aea11bdf716c51d4dc753a79b83ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3058, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3058, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_73cb5ec03c7a56bb73c2a4c46c2fccda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b66aea11bdf716c51d4dc753a79b83ce
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73cb5ec03c7a56bb73c2a4c46c2fccda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b66aea11bdf716c51d4dc753a79b83ce
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73cb5ec03c7a56bb73c2a4c46c2fccda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b66aea11bdf716c51d4dc753a79b83ce
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73cb5ec03c7a56bb73c2a4c46c2fccda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b66aea11bdf716c51d4dc753a79b83ce
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5305e70b08625b41553a79718f2662bd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3793, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3793, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a2706cdeae57e5a2cdf651f401fd7308(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5305e70b08625b41553a79718f2662bd
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2706cdeae57e5a2cdf651f401fd7308(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5305e70b08625b41553a79718f2662bd
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2706cdeae57e5a2cdf651f401fd7308(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5305e70b08625b41553a79718f2662bd
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2706cdeae57e5a2cdf651f401fd7308(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5305e70b08625b41553a79718f2662bd
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3ea942f59bdaf3924bcf57deaa7a0dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df01a60f25e84fe7f62858a629f443f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3ea942f59bdaf3924bcf57deaa7a0dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df01a60f25e84fe7f62858a629f443f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a38f701468c986ce2faf8857962b117b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11390042304992676], [0.4904465973377228], [0.13921834528446198], [0.496462345123291]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.19393745064735413], [0.1851010024547577], [0.08248813450336456], [0.12678833305835724]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_8eca232f959791d3bb9ed417cc048b82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3694194257259369], [0.4472231864929199], [0.41353461146354675], [0.07219047099351883]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.020291708409786224], [0.017377078533172607], [0.2168794572353363], [0.03630860522389412]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_a9dd42a2e3f5b32e3ae1f6022763737b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06521501392126083], [0.034968309104442596], [0.26277071237564087], [0.01609121635556221]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.339884489774704], [0.06699070334434509], [0.30670201778411865], [0.038516294211149216]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_74ed3351259a6804f2cfc8fe5244e077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17842870950698853], [0.3201243281364441], [0.30915895104408264], [0.2734312415122986]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3211865723133087], [0.02025376260280609], [0.31398555636405945], [0.023009276017546654]], dtype='float32').reshape([4, 1]),
        ]


class PrimitiveOp_02daeebd033f32ccce7772ba8881a385(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2042, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2042, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0a5854fd163a769af37f5804fcb40159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02daeebd033f32ccce7772ba8881a385
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a5854fd163a769af37f5804fcb40159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02daeebd033f32ccce7772ba8881a385
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a5854fd163a769af37f5804fcb40159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02daeebd033f32ccce7772ba8881a385
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a5854fd163a769af37f5804fcb40159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02daeebd033f32ccce7772ba8881a385
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_432df3797e5f57023b82fab6a50fa7e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a860fac71210374cca5ca09f1371af03
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_432df3797e5f57023b82fab6a50fa7e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a860fac71210374cca5ca09f1371af03
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_84d5eb269bfd34f9bc32ecb9f4891a02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2850aeb421d78f007bb2e5ce26a7c30c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_84d5eb269bfd34f9bc32ecb9f4891a02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2850aeb421d78f007bb2e5ce26a7c30c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0b74dfbe80ca4eee035b1a8a60be23b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02b99da8b743338a1130415f700d3737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b74dfbe80ca4eee035b1a8a60be23b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02b99da8b743338a1130415f700d3737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b74dfbe80ca4eee035b1a8a60be23b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1daa0eab58f1c8cb5cd5f4a55da4ee94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4185, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_558968234ea1f1c044750871f01d27ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1daa0eab58f1c8cb5cd5f4a55da4ee94
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_558968234ea1f1c044750871f01d27ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1daa0eab58f1c8cb5cd5f4a55da4ee94
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_558968234ea1f1c044750871f01d27ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1daa0eab58f1c8cb5cd5f4a55da4ee94
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_558968234ea1f1c044750871f01d27ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1daa0eab58f1c8cb5cd5f4a55da4ee94
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02b99da8b743338a1130415f700d3737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b74dfbe80ca4eee035b1a8a60be23b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02b99da8b743338a1130415f700d3737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b74dfbe80ca4eee035b1a8a60be23b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()