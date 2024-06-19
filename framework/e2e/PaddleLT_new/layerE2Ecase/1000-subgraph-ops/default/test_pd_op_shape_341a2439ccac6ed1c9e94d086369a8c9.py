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



class PrimitiveOp_0af5f77c4503df5395cfb1f24d9ad8d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7747d8a8d093150b548e541252be6983(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0af5f77c4503df5395cfb1f24d9ad8d6
    def get_inputs(self):
        return [
            paddle.uniform([1508, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ff7a6bd9119c459b94ede2937065347b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8203f0479473c3849b36ffdd8220d6be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8203f0479473c3849b36ffdd8220d6be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1194ecbb14f91d7613a1d19e381f411f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0af5f77c4503df5395cfb1f24d9ad8d6
    def get_inputs(self):
        return [
            paddle.uniform([2377, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fe88c545bc9bf1fa03fe577d28d117f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fe88c545bc9bf1fa03fe577d28d117f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_62b160ddd8135921d542ef6232fcc812(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_61d137ca2b74d66b6752ab4daf587fa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b160ddd8135921d542ef6232fcc812
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5016d19c3a4bc1a119dcc864157862b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_032f7b7cef3d8d2a9488cb32493d659e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5016d19c3a4bc1a119dcc864157862b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_17a237bea2bc7bb241e953709d96b284(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 128, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fa7fc611d3b1c188c6710e5c5b858528(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17a237bea2bc7bb241e953709d96b284
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fa7fc611d3b1c188c6710e5c5b858528(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17a237bea2bc7bb241e953709d96b284
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_151b63b7dfdcabb86a745bc0661536c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f23b4492af6a8d66bacd384bd49c3b3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_151b63b7dfdcabb86a745bc0661536c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ea45f2e64ecff4d37ec12365ec375f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0af5f77c4503df5395cfb1f24d9ad8d6
    def get_inputs(self):
        return [
            paddle.uniform([2015, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f8a629fd73c39f740169dbec435c71d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f8a629fd73c39f740169dbec435c71d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_14186721dfb718b18893dfb75e22925f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e6a57ed2e26e198b8ae23671cb8076eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14186721dfb718b18893dfb75e22925f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_feccf05c973a33e3366f8ce625527f4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 256, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_524dd5df8013c04ecff4667421076e59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feccf05c973a33e3366f8ce625527f4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f78aecce4ce547e491860212e5feaaba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0af5f77c4503df5395cfb1f24d9ad8d6
    def get_inputs(self):
        return [
            paddle.uniform([1830, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_53018b57cb509fbe36a3299b64e746d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_53018b57cb509fbe36a3299b64e746d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_eea06d007e5ffd76a9251e78fc212b52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d52d5fcb036460b6b40a90f6bcffa75f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eea06d007e5ffd76a9251e78fc212b52
    def get_inputs(self):
        return [
            paddle.uniform([4850], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_399028c5ac6d5e9fffa9c15f97235d1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0af5f77c4503df5395cfb1f24d9ad8d6
    def get_inputs(self):
        return [
            paddle.uniform([3039, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a843d7a1c71af23f7916ae0de01c2863(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a843d7a1c71af23f7916ae0de01c2863(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5e512e78c737536e57f11c363faf5464(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59b7f1f00788db90010f472bfedf6627(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e512e78c737536e57f11c363faf5464
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5cbb991d642ed2d1045a8916a84ba49d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0af5f77c4503df5395cfb1f24d9ad8d6
    def get_inputs(self):
        return [
            paddle.uniform([2046, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7418ec91c9ed752cb01a9109dfc7f64e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7418ec91c9ed752cb01a9109dfc7f64e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7265aa2b813a7568191de0b52173e64d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eea06d007e5ffd76a9251e78fc212b52
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.9166666865348816, -0.8333333134651184, -0.75, -0.6666666865348816, -0.5833333134651184, -0.5, -0.4166666567325592, -0.3333333432674408, -0.25, -0.1666666716337204, -0.0833333358168602, 5.551115123125783e-17, 0.0833333358168602, 0.1666666716337204, 0.25, 0.3333333432674408, 0.4166666567325592, 0.5, 0.5833333134651184, 0.6666666865348816, 0.75, 0.8333333134651184, 0.9166666865348816, 1.0], dtype='float32').reshape([25]),
        ]


class TestPrimitiveOp_f05dc9315c5e2815f17b090a6ee9c02a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eea06d007e5ffd76a9251e78fc212b52
    def get_inputs(self):
        return [
            paddle.uniform([38], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_22b6cbd2df6c13f81bc6715794e27c08(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_da91a6c9784d8377d125558ae714941b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22b6cbd2df6c13f81bc6715794e27c08
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_61d137ca2b74d66b6752ab4daf587fa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b160ddd8135921d542ef6232fcc812
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_61d137ca2b74d66b6752ab4daf587fa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b160ddd8135921d542ef6232fcc812
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_61d137ca2b74d66b6752ab4daf587fa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b160ddd8135921d542ef6232fcc812
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_17eb3011c7666d621151d484503d4aad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c1c796944822fc0ad1d1a8b16934ef47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17eb3011c7666d621151d484503d4aad
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_da91a6c9784d8377d125558ae714941b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22b6cbd2df6c13f81bc6715794e27c08
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_da91a6c9784d8377d125558ae714941b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22b6cbd2df6c13f81bc6715794e27c08
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_da91a6c9784d8377d125558ae714941b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22b6cbd2df6c13f81bc6715794e27c08
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0397f98ff704ec7871227f2db7d422e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e05ef251d4232b435d798f89510d8eca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0397f98ff704ec7871227f2db7d422e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f2fc91cb8a0fa49517a6e610c386eb20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0af5f77c4503df5395cfb1f24d9ad8d6
    def get_inputs(self):
        return [
            paddle.uniform([5498, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_29f7ba8530ce3795380fb6811371a033(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_29f7ba8530ce3795380fb6811371a033(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_78a49e43d146fa80c646c849b4293ad4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0af5f77c4503df5395cfb1f24d9ad8d6
    def get_inputs(self):
        return [
            paddle.uniform([1074, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8b5b2125ceb79892100e7a821e3a456b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8b5b2125ceb79892100e7a821e3a456b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_59b7f1f00788db90010f472bfedf6627(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e512e78c737536e57f11c363faf5464
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be2f26dba8373fbdcad7e0197e25a6fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eea06d007e5ffd76a9251e78fc212b52
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.8947368264198303, -0.7894737124443054, -0.6842105388641357, -0.5789473652839661, -0.4736842215061188, -0.3684210479259491, -0.2631579041481018, -0.15789473056793213, -0.05263157933950424, 0.05263157933950424, 0.15789473056793213, 0.2631579041481018, 0.3684210479259491, 0.4736842215061188, 0.5789473652839661, 0.6842105388641357, 0.7894737124443054, 0.8947368264198303, 1.0], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_5fd2118522bae2f3997af0448b7db7db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eea06d007e5ffd76a9251e78fc212b52
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.931034505367279, -0.8620689511299133, -0.7931034564971924, -0.7241379022598267, -0.6551724076271057, -0.5862069129943848, -0.517241358757019, -0.4482758641242981, -0.37931033968925476, -0.3103448152542114, -0.24137930572032928, -0.17241379618644714, -0.1034482792019844, -0.03448275849223137, 0.03448275849223137, 0.1034482792019844, 0.17241379618644714, 0.24137930572032928, 0.3103448152542114, 0.37931033968925476, 0.4482758641242981, 0.517241358757019, 0.5862069129943848, 0.6551724076271057, 0.7241379022598267, 0.7931034564971924, 0.8620689511299133, 0.931034505367279, 1.0], dtype='float32').reshape([30]),
        ]


class PrimitiveOp_eafef35825776602f95a14b7b3bfba0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 97, 97], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_35ef191d3dd6342a9aafc1df49eea869(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eafef35825776602f95a14b7b3bfba0b
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c9b6d7fa75e64cbe1cc30ffe65eff71d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0af5f77c4503df5395cfb1f24d9ad8d6
    def get_inputs(self):
        return [
            paddle.uniform([1773, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_91c2a3a950cd8f1a1e1b9b039b9d4771(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_91c2a3a950cd8f1a1e1b9b039b9d4771(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a5924548c505b3f6eb8227b4ba7fcebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eea06d007e5ffd76a9251e78fc212b52
    def get_inputs(self):
        return [
            paddle.uniform([17524], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_27704273fea8638c221f3aaef3cab5d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eea06d007e5ffd76a9251e78fc212b52
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.8571428656578064, -0.7142857313156128, -0.5714285969734192, -0.4285714328289032, -0.2857142984867096, -0.1428571492433548, 5.551115123125783e-17, 0.1428571492433548, 0.2857142984867096, 0.4285714328289032, 0.5714285969734192, 0.7142857313156128, 0.8571428656578064, 1.0], dtype='float32').reshape([15]),
        ]


class TestPrimitiveOp_7265aa2b813a7568191de0b52173e64d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eea06d007e5ffd76a9251e78fc212b52
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.9166666865348816, -0.8333333134651184, -0.75, -0.6666666865348816, -0.5833333134651184, -0.5, -0.4166666567325592, -0.3333333432674408, -0.25, -0.1666666716337204, -0.0833333358168602, 5.551115123125783e-17, 0.0833333358168602, 0.1666666716337204, 0.25, 0.3333333432674408, 0.4166666567325592, 0.5, 0.5833333134651184, 0.6666666865348816, 0.75, 0.8333333134651184, 0.9166666865348816, 1.0], dtype='float32').reshape([25]),
        ]


class TestPrimitiveOp_9da8e1857f1f8c837d1c8626af90fd1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0af5f77c4503df5395cfb1f24d9ad8d6
    def get_inputs(self):
        return [
            paddle.uniform([4224, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bdcded5830fe4e0cfaad28b761392b02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bdcded5830fe4e0cfaad28b761392b02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cf1d4e5e7c1d3e494fc549897dd2f608(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ccb8760ce3afaec19a0ddf689e35c4f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf1d4e5e7c1d3e494fc549897dd2f608
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4972515e9455844480dc7f21fa46b165(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eea06d007e5ffd76a9251e78fc212b52
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.9130434989929199, -0.8260869383811951, -0.739130437374115, -0.6521739363670349, -0.5652173757553101, -0.47826087474823, -0.3913043439388275, -0.30434781312942505, -0.21739129722118378, -0.1304347813129425, -0.043478261679410934, 0.043478261679410934, 0.1304347813129425, 0.21739129722118378, 0.30434781312942505, 0.3913043439388275, 0.47826087474823, 0.5652173757553101, 0.6521739363670349, 0.739130437374115, 0.8260869383811951, 0.9130434989929199, 1.0], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_3b9a639f2f4da6dcd26479370373c830(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eea06d007e5ffd76a9251e78fc212b52
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ea6d972aabafd7cca5c63a3d0e23093e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0af5f77c4503df5395cfb1f24d9ad8d6
    def get_inputs(self):
        return [
            paddle.uniform([4657, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2c3f320017c2f3979908ebf99ce2ebce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2c3f320017c2f3979908ebf99ce2ebce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_48d30a4ebacacb85379437c939f9107d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0af5f77c4503df5395cfb1f24d9ad8d6
    def get_inputs(self):
        return [
            paddle.uniform([3770, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_755db93236c464eee334fd5e0f8bf9f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_755db93236c464eee334fd5e0f8bf9f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff7a6bd9119c459b94ede2937065347b
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()