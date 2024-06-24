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


class PrimitiveOp_e1f1fa59b26fcd2de82862df3c9dc7b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1758, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1758, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1280c253548347ccb6147e69cc54ecb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1f1fa59b26fcd2de82862df3c9dc7b3
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1280c253548347ccb6147e69cc54ecb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1f1fa59b26fcd2de82862df3c9dc7b3
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1280c253548347ccb6147e69cc54ecb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1f1fa59b26fcd2de82862df3c9dc7b3
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1280c253548347ccb6147e69cc54ecb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1f1fa59b26fcd2de82862df3c9dc7b3
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_63c7a7374e854571294cc446dbf2aa02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11550065129995346], [0.374041348695755], [0.15645268559455872], [0.2919510304927826], [0.16898958384990692], [0.05408820882439613], [0.11748014390468597], [0.40530943870544434], [0.1729610115289688]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3067001700401306], [0.2585631310939789], [0.4714326858520508], [0.13411420583724976], [0.47448375821113586], [0.4214091897010803], [0.38982850313186646], [0.18900315463542938], [0.18645043671131134]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_036d93e435bec160d8f0c8354ea049c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.044602248817682266], [0.11456053704023361], [0.33949682116508484], [0.03393561393022537], [0.4222550392150879], [0.029572542756795883], [0.15619488060474396], [0.0473758690059185], [0.3221006393432617]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.14979442954063416], [0.44975969195365906], [0.3147315979003906], [0.0915440171957016], [0.21444787085056305], [0.03469356149435043], [0.23302797973155975], [0.32175347208976746], [0.4466075897216797]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_32a90e7e70f817253ee35ac5d878072c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4423205256462097], [0.18592821061611176], [0.2645948827266693], [0.24683165550231934], [0.23438315093517303], [0.4034620225429535], [0.20104055106639862], [0.18984976410865784], [0.18868638575077057]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.12630529701709747], [0.4453306794166565], [0.027548898011446], [0.3977745771408081], [0.25538399815559387], [0.17480947077274323], [0.2219296097755432], [0.3580722510814667], [0.44802966713905334]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_c3237871455ad656a0444e1daa661f14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.24950331449508667], [0.34855425357818604], [0.26499083638191223], [0.0974547415971756], [0.49846959114074707], [0.30478185415267944], [0.2055416703224182], [0.3756718039512634], [0.030764833092689514]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.41259145736694336], [0.3572691082954407], [0.1078881248831749], [0.3949889540672302], [0.296988844871521], [0.4767949879169464], [0.15632693469524384], [0.31878870725631714], [0.17685101926326752]], dtype='float32').reshape([9, 1]),
        ]


class PrimitiveOp_4dcf4d367c48589c0351874738ee1901(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5593, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5593, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_98d71763afd0c9151fa8036a4f64aaa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dcf4d367c48589c0351874738ee1901
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98d71763afd0c9151fa8036a4f64aaa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dcf4d367c48589c0351874738ee1901
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98d71763afd0c9151fa8036a4f64aaa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dcf4d367c48589c0351874738ee1901
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98d71763afd0c9151fa8036a4f64aaa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dcf4d367c48589c0351874738ee1901
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_07fc09735996496e6e7119f8c04be5c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6f51e0ff307ce9e9ae0a6d3c1cc52e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44525033235549927, 0.4553702473640442, 0.3241686224937439, 0.27912789583206177, 0.4798333942890167, 0.1506512463092804], dtype='float32').reshape([6]),
            paddle.to_tensor([0.18762415647506714, 0.2747806906700134, 0.4200975298881531, 0.3786199986934662, 0.03297353908419609, 0.23518399894237518], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_55fbb8d24e160d03260e204b7df336ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6f51e0ff307ce9e9ae0a6d3c1cc52e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.40655165910720825, 0.2496812641620636, 0.49778977036476135, 0.06130501255393028, 0.47947970032691956, 0.3735698163509369], dtype='float32').reshape([6]),
            paddle.to_tensor([0.18261484801769257, 0.3417728841304779, 0.2622556686401367, 0.1527484655380249, 0.36947619915008545, 0.32382044196128845], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_a73fb62d6d57a53f1e22c6ec251b7eec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6f51e0ff307ce9e9ae0a6d3c1cc52e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2925397753715515, 0.4553702473640442, 0.23048850893974304, 0.27912789583206177, 0.4798333942890167, 0.1506512463092804], dtype='float32').reshape([6]),
            paddle.to_tensor([0.28722062706947327, 0.21731960773468018, 0.3951013386249542, 0.3923845887184143, 0.3277442157268524, 0.35763120651245117], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_5d64f13eda67e444c5a516512c3a0731(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6f51e0ff307ce9e9ae0a6d3c1cc52e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.40655165910720825, 0.18528297543525696, 0.49778977036476135, 0.06130501255393028, 0.47947970032691956, 0.3735698163509369], dtype='float32').reshape([6]),
            paddle.to_tensor([0.43829435110092163, 0.012103257700800896, 0.42449864745140076, 0.4616372585296631, 0.0976329892873764, 0.0438818633556366], dtype='float32').reshape([6]),
        ]


class PrimitiveOp_87451e95011559cf2df6f205c51db6e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1763, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1763, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4f685122762ec22e6eceaf76a52369c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87451e95011559cf2df6f205c51db6e0
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f685122762ec22e6eceaf76a52369c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87451e95011559cf2df6f205c51db6e0
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f685122762ec22e6eceaf76a52369c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87451e95011559cf2df6f205c51db6e0
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f685122762ec22e6eceaf76a52369c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87451e95011559cf2df6f205c51db6e0
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_bc0437ade14c711de53fa835b2a334f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.36167198419570923]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3290674686431885]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_df480b34578a4ac08cb6861ff68ed8e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1334301382303238]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0780901163816452]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_2138405c86c7be39c6f9ec3618b9f361(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.491269052028656]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.4348372220993042]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_443e20d7d1127363aaa3b2f941847c26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09417319297790527]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.09474091231822968]], dtype='float32').reshape([1, 1]),
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


class TestPrimitiveOp_018b7d4093111a455394b0c617d7fc1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.40875673294067383], [0.17066887021064758], [0.32239457964897156], [0.12811346352100372], [0.056965190917253494], [0.37778374552726746]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3079056143760681], [0.3012850880622864], [0.45780348777770996], [0.38056784868240356], [0.45184314250946045], [0.28739556670188904]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_0c02879b93ec20654db1f0c0a9517258(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14837408065795898], [0.1733187437057495], [0.2672334313392639], [0.09106913208961487], [0.15825320780277252], [0.12337625026702881]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.23708929121494293], [0.3412092924118042], [0.4712662696838379], [0.33640754222869873], [0.018445204943418503], [0.33481597900390625]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_834728b9fcfa89fbea3d37127cfe809f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.49020639061927795], [0.06694857776165009], [0.37551629543304443], [0.16441957652568817], [0.056271616369485855], [0.4976474642753601]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.03940660506486893], [0.4823766052722931], [0.3955044746398926], [0.22639241814613342], [0.3185456097126007], [0.04834653064608574]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_5ebc865e6bfd931f51bb7b6b5b7efffe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.38628971576690674], [0.3818877637386322], [0.4154316186904907], [0.09654207527637482], [0.48540663719177246], [0.23541131615638733]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.13275480270385742], [0.24877241253852844], [0.14286577701568604], [0.02221701852977276], [0.3493429720401764], [0.074555404484272]], dtype='float32').reshape([6, 1]),
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


class PrimitiveOp_8e2542ecc7eb22f1156c725f1f246744(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2076, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2076, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ce084b021c2bca52332334dcb1e9cd93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e2542ecc7eb22f1156c725f1f246744
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce084b021c2bca52332334dcb1e9cd93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e2542ecc7eb22f1156c725f1f246744
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce084b021c2bca52332334dcb1e9cd93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e2542ecc7eb22f1156c725f1f246744
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce084b021c2bca52332334dcb1e9cd93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e2542ecc7eb22f1156c725f1f246744
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_cf3cba04f4b73197535ffa59b215a4b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4642, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4642, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9a4ad71a7065f43162da9b82a3912437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf3cba04f4b73197535ffa59b215a4b2
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a4ad71a7065f43162da9b82a3912437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf3cba04f4b73197535ffa59b215a4b2
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a4ad71a7065f43162da9b82a3912437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf3cba04f4b73197535ffa59b215a4b2
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a4ad71a7065f43162da9b82a3912437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf3cba04f4b73197535ffa59b215a4b2
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_97e6fd284670ccfacc176917affaa5b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1047, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1047, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b014c5fb5e98e9cbb710634a5c5d45be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97e6fd284670ccfacc176917affaa5b3
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b014c5fb5e98e9cbb710634a5c5d45be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97e6fd284670ccfacc176917affaa5b3
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b014c5fb5e98e9cbb710634a5c5d45be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97e6fd284670ccfacc176917affaa5b3
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b014c5fb5e98e9cbb710634a5c5d45be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97e6fd284670ccfacc176917affaa5b3
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_9fd550c740fc7934ef01940a6472e115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12497831881046295], [0.2808596193790436], [0.11296632140874863], [0.20338411629199982], [0.2920495271682739]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.39957284927368164], [0.406246155500412], [0.24995972216129303], [0.3918718695640564], [0.4563639461994171]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_ead29d6201dd80224d8f9eb26f1c04fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4368440806865692], [0.25516924262046814], [0.033184755593538284], [0.33212995529174805], [0.40790942311286926]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.47180286049842834], [0.31962135434150696], [0.32587990164756775], [0.07404229044914246], [0.4006950557231903]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_b336c98d64328ff31b512fb7e489f5e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15261472761631012], [0.34060344099998474], [0.2098308652639389], [0.34933146834373474], [0.20875117182731628]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.28228238224983215], [0.4601689577102661], [0.23982450366020203], [0.4331777095794678], [0.05244790017604828]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_0e77373f92e936c477a1fb1689e6c3c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05367068201303482], [0.36561113595962524], [0.4665828347206116], [0.2871648371219635], [0.07508329302072525]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.4062413275241852], [0.011468961834907532], [0.3539048135280609], [0.33214429020881653], [0.37686073780059814]], dtype='float32').reshape([5, 1]),
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


class PrimitiveOp_929da48edb686b696161696d6ceee34d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2359, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2359, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b38c03042c353b3b523c0b386438f0b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_929da48edb686b696161696d6ceee34d
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b38c03042c353b3b523c0b386438f0b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_929da48edb686b696161696d6ceee34d
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b38c03042c353b3b523c0b386438f0b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_929da48edb686b696161696d6ceee34d
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b38c03042c353b3b523c0b386438f0b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_929da48edb686b696161696d6ceee34d
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6d10d3d59f2b33d7891cc2735e1aa058(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3049, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3049, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_11ed2f649f540eafaefe1621060469f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d10d3d59f2b33d7891cc2735e1aa058
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11ed2f649f540eafaefe1621060469f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d10d3d59f2b33d7891cc2735e1aa058
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11ed2f649f540eafaefe1621060469f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d10d3d59f2b33d7891cc2735e1aa058
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11ed2f649f540eafaefe1621060469f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d10d3d59f2b33d7891cc2735e1aa058
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bd76c817c6e36565bc4db38d3ad2a666(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3806, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3806, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b25ce733722330058be780b28ebd94df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd76c817c6e36565bc4db38d3ad2a666
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b25ce733722330058be780b28ebd94df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd76c817c6e36565bc4db38d3ad2a666
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b25ce733722330058be780b28ebd94df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd76c817c6e36565bc4db38d3ad2a666
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b25ce733722330058be780b28ebd94df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd76c817c6e36565bc4db38d3ad2a666
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_fd9b1458874b1f4a821da9662601524b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20518168807029724], [0.4319309592247009], [0.47744256258010864], [0.42946767807006836]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.16607213020324707], [0.1927356868982315], [0.15653924643993378], [0.1810394525527954]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_1ee32bf7b23b0fdeda336e1d9282886e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.48733291029930115], [0.3396559953689575], [0.02164270356297493], [0.22589930891990662]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.05042451620101929], [0.23747287690639496], [0.43582987785339355], [0.027551813051104546]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_c94aab8d19b1876e036ad8f6ced4fa3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0391080342233181], [0.06995806097984314], [0.4809706211090088], [0.22571998834609985]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.15882641077041626], [0.048668403178453445], [0.17359651625156403], [0.05119822919368744]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_a5aa3de6557cddb67e4ff015eedceb01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07258392870426178], [0.43579286336898804], [0.34903889894485474], [0.37684279680252075]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.43369781970977783], [0.18939900398254395], [0.3099403977394104], [0.4193747043609619]], dtype='float32').reshape([4, 1]),
        ]


class PrimitiveOp_322dc7d222a0ded41d8db44259ec32fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2054, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2054, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0aa9fcdba35a22eb16157585996b0230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_322dc7d222a0ded41d8db44259ec32fc
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0aa9fcdba35a22eb16157585996b0230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_322dc7d222a0ded41d8db44259ec32fc
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0aa9fcdba35a22eb16157585996b0230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_322dc7d222a0ded41d8db44259ec32fc
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0aa9fcdba35a22eb16157585996b0230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_322dc7d222a0ded41d8db44259ec32fc
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_2988ef22bf466aadda29aee91b4ddaca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4218, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4218, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cbe22ce865fefd9221a84e3de83fd199(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2988ef22bf466aadda29aee91b4ddaca
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbe22ce865fefd9221a84e3de83fd199(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2988ef22bf466aadda29aee91b4ddaca
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbe22ce865fefd9221a84e3de83fd199(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2988ef22bf466aadda29aee91b4ddaca
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbe22ce865fefd9221a84e3de83fd199(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2988ef22bf466aadda29aee91b4ddaca
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
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