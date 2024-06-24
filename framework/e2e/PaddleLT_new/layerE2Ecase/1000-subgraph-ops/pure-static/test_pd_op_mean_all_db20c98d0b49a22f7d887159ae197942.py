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



class PrimitiveOp_73e799f11a08ee0bcc474abac15e2d9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6ce7e8d20fc4d34b4cb888cd4132ee1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73e799f11a08ee0bcc474abac15e2d9b
    def get_inputs(self):
        return [
            paddle.uniform([4, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_62a5de42bf478e259aab45f639dca144(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3d067fbc39bc619889d50242ea8791f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62a5de42bf478e259aab45f639dca144
    def get_inputs(self):
        return [
            paddle.uniform([3, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e27fd4f19d39cc2fae0e64f2ffd7f64e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1786, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a3e9a684df7cab83f4f4bd2798e68036(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e27fd4f19d39cc2fae0e64f2ffd7f64e
    def get_inputs(self):
        return [
            paddle.uniform([1786, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4c66901e27a013ba4803109a638ac472(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5529, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b3e5cd5c32fffdd72b82b64c3a7759b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c66901e27a013ba4803109a638ac472
    def get_inputs(self):
        return [
            paddle.uniform([5529, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0be9d5a910b19e133c3e5f7b46e337ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1767, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f555833dee497024523e692e6a17def5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0be9d5a910b19e133c3e5f7b46e337ed
    def get_inputs(self):
        return [
            paddle.uniform([1767, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fe7e927cdf586a5e0844b3a54bd79424(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1490, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b8f2e5586c4605b737845787dc17c29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe7e927cdf586a5e0844b3a54bd79424
    def get_inputs(self):
        return [
            paddle.uniform([1490, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_204ba885d2c69290f2066f72b4fcd51c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2010, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_89d548336825d1fb191e8738ad16b853(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204ba885d2c69290f2066f72b4fcd51c
    def get_inputs(self):
        return [
            paddle.uniform([2010, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9fad7c13297dbc411c648f4c3a56101a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4663, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_233ff222c379541ac89f07e17f348ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fad7c13297dbc411c648f4c3a56101a
    def get_inputs(self):
        return [
            paddle.uniform([4663, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f0f6b48512914285ca1fbf6b8f068334(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8e0c66dbf000b9d69be376ce6190e19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0f6b48512914285ca1fbf6b8f068334
    def get_inputs(self):
        return [
            paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f3d83aba74f562b4e3a5c8a72302c300(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1090, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_acee3b11e1e6ce62964ab69397dc2fae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3d83aba74f562b4e3a5c8a72302c300
    def get_inputs(self):
        return [
            paddle.uniform([1090, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fde98cd67fe5dcded666570e72d869f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2374, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cb8c767629c2dd52a542603a6ccea248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fde98cd67fe5dcded666570e72d869f8
    def get_inputs(self):
        return [
            paddle.uniform([2374, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_85350c77cf6133abacb4dc2eea8c1392(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3058, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_96a313ce0c803ed4134f2dfd75679716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85350c77cf6133abacb4dc2eea8c1392
    def get_inputs(self):
        return [
            paddle.uniform([3058, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4cd53772b2e3688800b0b5bb8ea23452(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3793, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e412fd2d3f4101623b5d5db10e242f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cd53772b2e3688800b0b5bb8ea23452
    def get_inputs(self):
        return [
            paddle.uniform([3793, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_64e8b480d6c6ff97bb12729c51d14469(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3014b352d7b5f68dbb59af9086b1a87a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64e8b480d6c6ff97bb12729c51d14469
    def get_inputs(self):
        return [
            paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9578df6a0008fc40f546814421a35323(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2042, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cebb27e4c14b8563e4f7954d72273005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9578df6a0008fc40f546814421a35323
    def get_inputs(self):
        return [
            paddle.uniform([2042, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_78d340f22d2cc2544489ccb4d7584152(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_67790ba1ccedd181085ba323a65f9601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78d340f22d2cc2544489ccb4d7584152
    def get_inputs(self):
        return [
            paddle.uniform([4185, 4], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()