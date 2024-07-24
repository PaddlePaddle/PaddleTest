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


class PrimitiveOp_2dd2a51971de8e2053cccb201d865675(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1787, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1787, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_780e2fe5bae6736cc570f771afc43bb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2dd2a51971de8e2053cccb201d865675
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_780e2fe5bae6736cc570f771afc43bb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2dd2a51971de8e2053cccb201d865675
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_780e2fe5bae6736cc570f771afc43bb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2dd2a51971de8e2053cccb201d865675
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_780e2fe5bae6736cc570f771afc43bb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2dd2a51971de8e2053cccb201d865675
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_b3dfacc9ab399b8c245c9783353f3649(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39222896099090576], [0.4460776746273041], [0.2511337995529175], [0.4043271541595459], [0.4428388476371765], [0.019881412386894226], [0.4657576084136963], [0.4012521505355835], [0.40715867280960083]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3243171274662018], [0.49249011278152466], [0.2755092680454254], [0.2209128737449646], [0.23234640061855316], [0.21800071001052856], [0.1990540474653244], [0.06053031235933304], [0.49496304988861084]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_49e16e8b14e6b0e58c9559e6a9387e51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3726502060890198], [0.35267090797424316], [0.026435574516654015], [0.40623462200164795], [0.3046559691429138], [0.4644503593444824], [0.2115304172039032], [0.4490237534046173], [0.2872993052005768]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.45809561014175415], [0.0487419031560421], [0.27907347679138184], [0.3332546353340149], [0.025723274797201157], [0.2000621259212494], [0.2891344726085663], [0.06198172643780708], [0.0746815949678421]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_2981a4ab7c5aec5a53c5313febc50180(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.340059757232666], [0.387004017829895], [0.4251270294189453], [0.4898195266723633], [0.31481853127479553], [0.4277181327342987], [0.0343017652630806], [0.3381534516811371], [0.09708041697740555]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.14785392582416534], [0.2970240116119385], [0.49885281920433044], [0.40843749046325684], [0.3837003707885742], [0.23476886749267578], [0.007210989482700825], [0.48448464274406433], [0.25622862577438354]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_e3d8fcaf59a888e942d390239d7397d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.29957494139671326], [0.36993101239204407], [0.43416088819503784], [0.06923043727874756], [0.01820189692080021], [0.030403364449739456], [0.3196682929992676], [0.24978111684322357], [0.1863652914762497]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.08313236385583878], [0.07048333436250687], [0.31889981031417847], [0.3709265887737274], [0.33858680725097656], [0.4856625199317932], [0.4982873797416687], [0.249019056558609], [0.03643830493092537]], dtype='float32').reshape([9, 1]),
        ]


class PrimitiveOp_adad483197e83c8b916dbc62be95aad3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5585, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5585, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55d547f9b303cf12e5539bb663c95194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adad483197e83c8b916dbc62be95aad3
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55d547f9b303cf12e5539bb663c95194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adad483197e83c8b916dbc62be95aad3
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55d547f9b303cf12e5539bb663c95194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adad483197e83c8b916dbc62be95aad3
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55d547f9b303cf12e5539bb663c95194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adad483197e83c8b916dbc62be95aad3
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_a0669964af4fa5dbc96dc3b8eadee176(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6f51e0ff307ce9e9ae0a6d3c1cc52e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.446471244096756, 0.4455740749835968, 0.4889226257801056, 0.39720895886421204, 0.35945671796798706, 0.4984632730484009], dtype='float32').reshape([6]),
            paddle.to_tensor([0.21949312090873718, 0.0012134052813053131, 0.4766252934932709, 0.11636532098054886, 0.2342921495437622, 0.4063635468482971], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_08ae773cce267fbfd7c8ce49d6d403f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6f51e0ff307ce9e9ae0a6d3c1cc52e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2992474138736725, 0.2878594994544983, 0.47377514839172363, 0.2895292341709137, 0.3072856068611145, 0.43330734968185425], dtype='float32').reshape([6]),
            paddle.to_tensor([0.031546175479888916, 0.1145879477262497, 0.332019567489624, 0.13446305692195892, 0.41516128182411194, 0.1730593889951706], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_1912be3d8f0e83fc8102ec56ca824028(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6f51e0ff307ce9e9ae0a6d3c1cc52e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.446471244096756, 0.14341187477111816, 0.4889226257801056, 0.025714602321386337, 0.20853398740291595, 0.15274621546268463], dtype='float32').reshape([6]),
            paddle.to_tensor([0.22107045352458954, 0.4040623903274536, 0.46366026997566223, 0.043019529432058334, 0.13830581307411194, 0.4344382584095001], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_57ae973f9f3e99343db74480cf33e11e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6f51e0ff307ce9e9ae0a6d3c1cc52e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2992474138736725, 0.2878594994544983, 0.47377514839172363, 0.05680272728204727, 0.043566759675741196, 0.43330734968185425], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3741452395915985, 0.11825495213270187, 0.027543269097805023, 0.20650611817836761, 0.1986602246761322, 0.02518189512193203], dtype='float32').reshape([6]),
        ]


class PrimitiveOp_ddbafd1565d1790437e01b7628fc5e4d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1774, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1774, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d696e90d7dd01cb0dcfd70f20162e178(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddbafd1565d1790437e01b7628fc5e4d
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d696e90d7dd01cb0dcfd70f20162e178(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddbafd1565d1790437e01b7628fc5e4d
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d696e90d7dd01cb0dcfd70f20162e178(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddbafd1565d1790437e01b7628fc5e4d
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d696e90d7dd01cb0dcfd70f20162e178(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddbafd1565d1790437e01b7628fc5e4d
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_85db5c6ea801b4f444b4031441971bb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1501, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1501, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_844aed400f6539ec248cd447a4a8379e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85db5c6ea801b4f444b4031441971bb7
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_844aed400f6539ec248cd447a4a8379e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85db5c6ea801b4f444b4031441971bb7
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_844aed400f6539ec248cd447a4a8379e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85db5c6ea801b4f444b4031441971bb7
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_844aed400f6539ec248cd447a4a8379e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85db5c6ea801b4f444b4031441971bb7
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_059f32081378a73cebb05668f884be13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.021477889269590378]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.4371948540210724]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_8bed67715a5645b3a71311d96b38a67a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.36230334639549255]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3298352062702179]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_fb0987bf07ca92ce947062fc82772ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.28265708684921265]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3178286850452423]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_dfd55b759781f82cbc3ccdf0a2f192e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06415817886590958]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.43161070346832275]], dtype='float32').reshape([1, 1]),
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


class TestPrimitiveOp_96c0c664d13212afa01cda94cbb715d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33286547660827637], [0.028109095990657806], [0.37021663784980774], [0.08470499515533447], [0.0046716793440282345], [0.19173109531402588]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3928629457950592], [0.3117378354072571], [0.14381171762943268], [0.40953099727630615], [0.0882832407951355], [0.15646472573280334]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_6900f710293e25ddbf8869cc38e7e89c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.26473021507263184], [0.4301255941390991], [0.009996733628213406], [0.43156635761260986], [0.38260313868522644], [0.05900372192263603]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.43764728307724], [0.4305558502674103], [0.3785059452056885], [0.4988960921764374], [0.02951250597834587], [0.22250328958034515]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_f7b4c55616f0ae75037cff49c4a22fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.35102102160453796], [0.4810973107814789], [0.48731809854507446], [0.059361398220062256], [0.40835464000701904], [0.47783151268959045]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.4764886200428009], [0.21417561173439026], [0.3296174705028534], [0.20144562423229218], [0.09571299701929092], [0.09185384213924408]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_c8af0052ef7828b462cf5486d1dd3c00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10640901327133179], [0.15210367739200592], [0.058579739183187485], [0.04452253505587578], [0.053740326315164566], [0.4524851441383362]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.29171887040138245], [0.3921818137168884], [0.2787969708442688], [0.39924129843711853], [0.09039394557476044], [0.48042625188827515]], dtype='float32').reshape([6, 1]),
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


class PrimitiveOp_8b484ce2e531586d2837766cdce3e626(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2049, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2049, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40b9223d04d242c0c11caee0ef5a477d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b484ce2e531586d2837766cdce3e626
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_40b9223d04d242c0c11caee0ef5a477d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b484ce2e531586d2837766cdce3e626
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_40b9223d04d242c0c11caee0ef5a477d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b484ce2e531586d2837766cdce3e626
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_40b9223d04d242c0c11caee0ef5a477d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b484ce2e531586d2837766cdce3e626
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_9c28c17ac754093db179404f8b3497cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4634, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4634, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3925f4708355cc26d16a88aeadb78d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c28c17ac754093db179404f8b3497cb
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3925f4708355cc26d16a88aeadb78d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c28c17ac754093db179404f8b3497cb
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3925f4708355cc26d16a88aeadb78d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c28c17ac754093db179404f8b3497cb
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3925f4708355cc26d16a88aeadb78d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c28c17ac754093db179404f8b3497cb
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_783f273c8b75b7ad103ceec73cdae47b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1000, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1000, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c7319def674a010153e323504e17a157(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_783f273c8b75b7ad103ceec73cdae47b
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7319def674a010153e323504e17a157(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_783f273c8b75b7ad103ceec73cdae47b
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7319def674a010153e323504e17a157(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_783f273c8b75b7ad103ceec73cdae47b
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7319def674a010153e323504e17a157(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_783f273c8b75b7ad103ceec73cdae47b
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_585477ea8c9b0be78c112715357cd248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3062000572681427], [0.3130226135253906], [0.17877483367919922], [0.19642746448516846], [0.023015061393380165]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.35707125067710876], [0.03344854712486267], [0.07517310976982117], [0.34104833006858826], [0.4732273817062378]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_bc4172ceda031b737186106ebc5ec3bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.02279740385711193], [0.34154266119003296], [0.30135348439216614], [0.4762178957462311], [0.05762705206871033]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.20563507080078125], [0.15860792994499207], [0.21742399036884308], [0.2578918933868408], [0.23720066249370575]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_deda4936a91f2182cd581d55d8cdbd21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10822069644927979], [0.06122198328375816], [0.35097381472587585], [0.4020307958126068], [0.36489900946617126]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.24273258447647095], [0.08545062690973282], [0.11754946410655975], [0.3820875585079193], [0.32701370120048523]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_f2efd0b6f53f685a7f864ce8fe88e659(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1466745138168335], [0.2202141284942627], [0.20410498976707458], [0.19355495274066925], [0.125055193901062]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.15814867615699768], [0.42982131242752075], [0.37481895089149475], [0.38867270946502686], [0.0035878224298357964]], dtype='float32').reshape([5, 1]),
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


class PrimitiveOp_674ff94218295339ec62e009c24e8220(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2382, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2382, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0a9b1cff5e1de37adfb68d079a08736a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_674ff94218295339ec62e009c24e8220
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a9b1cff5e1de37adfb68d079a08736a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_674ff94218295339ec62e009c24e8220
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a9b1cff5e1de37adfb68d079a08736a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_674ff94218295339ec62e009c24e8220
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a9b1cff5e1de37adfb68d079a08736a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_674ff94218295339ec62e009c24e8220
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8fe7c06b2ef56c07b29c2742ed58cda5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2976, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2976, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7cc11be0dc09fce388c8ab57bb829d58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fe7c06b2ef56c07b29c2742ed58cda5
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cc11be0dc09fce388c8ab57bb829d58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fe7c06b2ef56c07b29c2742ed58cda5
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cc11be0dc09fce388c8ab57bb829d58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fe7c06b2ef56c07b29c2742ed58cda5
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cc11be0dc09fce388c8ab57bb829d58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fe7c06b2ef56c07b29c2742ed58cda5
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d24bb8a8de4d75daffdac1d4bc162dd8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3753, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3753, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e395ea24e8cf0a9e0278ff3b10610d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d24bb8a8de4d75daffdac1d4bc162dd8
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e395ea24e8cf0a9e0278ff3b10610d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d24bb8a8de4d75daffdac1d4bc162dd8
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e395ea24e8cf0a9e0278ff3b10610d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d24bb8a8de4d75daffdac1d4bc162dd8
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e395ea24e8cf0a9e0278ff3b10610d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d24bb8a8de4d75daffdac1d4bc162dd8
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_51ceaa317f8818d263be53a2d5bfb8fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4963822662830353], [0.2475523203611374], [0.11298491805791855], [0.2765282988548279]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.25048601627349854], [0.43678945302963257], [0.09424278885126114], [0.2916932702064514]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_75630e57bc1be6cff46d058cdefdf39b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2910788357257843], [0.030457817018032074], [0.4478048086166382], [0.3236304521560669]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.1932741403579712], [0.1818334460258484], [0.12248531728982925], [0.3949283957481384]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_dab5a4ed530986c83d92c5059950cba5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17840133607387543], [0.15334481000900269], [0.35247161984443665], [0.10544773191213608]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3934442102909088], [0.42687827348709106], [0.24674589931964874], [0.20445282757282257]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_a202454f56bf69db7e69e91f6f616b8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.42456135153770447], [0.41002196073532104], [0.4354965388774872], [0.12312145531177521]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4565404951572418], [0.4529203772544861], [0.1775546818971634], [0.31460875272750854]], dtype='float32').reshape([4, 1]),
        ]


class PrimitiveOp_b7c10e8d11b00caa13e2ee7c4f71103f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1995, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1995, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_72e4353071c34cd17fa60ac0380280a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7c10e8d11b00caa13e2ee7c4f71103f
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72e4353071c34cd17fa60ac0380280a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7c10e8d11b00caa13e2ee7c4f71103f
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72e4353071c34cd17fa60ac0380280a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7c10e8d11b00caa13e2ee7c4f71103f
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72e4353071c34cd17fa60ac0380280a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7c10e8d11b00caa13e2ee7c4f71103f
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
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