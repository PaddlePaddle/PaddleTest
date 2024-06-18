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



class PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df90829d3ad22c9cc098c1be22401dc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 18, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_743078adaaebddb385daf13c0900f629(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_82ec862ab27380b36e1df75421d2d241(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 17, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_91c2f0d7925e81acb1cd8370d9c9f480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5c07c99dfb2d05e2e577ed855bbacc6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_913bbc17492eda43f19e34297207a7b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_891460f1c919862b4411f3f87e4b13b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 22, 22], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3dcf44bd33a2a1716488b2a76c729146(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f164ec9de70361ab73e7f3eb19eabd56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 44, 44], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dfd9059eab75a70d63e04f6ae2521cec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_07e993dc3cc7f15d90a6a440a657473b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d17b17b1ae194490c7acbfe58c92fc96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e8a375e2c38ff4ff395ab38700246621(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 16, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_33832cb6500de1d580319edd22f09eac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 17, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bdaaa38cc80ab96a5ec2876176a4f3e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([171, 480, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_babfe8aea69c4b2ca15088fca3e3cea9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_02cda3dd4738c73465cb1912fd687fda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_02cda3dd4738c73465cb1912fd687fda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_02b84f46be6c77ac328bc8ce832f60a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_02b84f46be6c77ac328bc8ce832f60a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d68636106c4021fb3e199f35654a9875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d68636106c4021fb3e199f35654a9875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bd4f00946401aeaaae259ef9b1ff81d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bd4f00946401aeaaae259ef9b1ff81d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c21ba27ba0454877fe9f72701b0d2afc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_41c464f616ffc83fc0f04a2c735518ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 320, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_93068ef490c51bab542f2ab7c9c6e2b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 44, 44], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ec7a7842710baf415ef90066caecbb68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([145, 60, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_ae9d183b612c41f511160504f9720dfb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6cf68f9f4b965aac6b18ee321a857017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_262fd8833014c159789a068aa2168314(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b774f9d291a2f6e81a61e28984d06409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0006d290c25dbd83885a4795f0d6e2c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dddbdc03a6b2963e93adfd4219b7ea89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_966904320c03579ce6f052ddbcca0988(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_64235219a68f97b8d01cc154246da979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ccfb701f461d854c3610c83e855ee1f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5c07c99dfb2d05e2e577ed855bbacc6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ccb03e3f03fc2c48006f2b99eefe034e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_20762ab85e63b911bfb75ab8d85cc22e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 88, 88], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cb3bdea54488eeee20b85100f3794126(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 9, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2c1814715664265b31c152d206652784(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9468a4940416e9cd05299ca035a7e919(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_05d31206c9a028a17f2caba43d98dd0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 68, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_70e00bc2e9f870afa6c66972382763e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6df8f3164d9d3e4f3aadc3547ea81640(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9c999f70f43324af9abb55fa4ac9cc04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_05051a690d786041d3a169708df9fa8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 22, 22], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d470e30c97b6d75921b86502bf9d0fb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_86ab4dc8dccc176e4791cef2b53d75f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_430e78eedf015a61507ef4adfaf43c0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_84a06bb7bcc661bd95b03d447979167e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 15, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a0fe952cd5524a5a2a79d2b188eda9cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 9, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_327e6a75b47e1baa74023d1cd3ef8860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1024, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9468a4940416e9cd05299ca035a7e919(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3c74773d25f44c3b8528e39af3a9e0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_48a0d5d244daf2ad1cf50fe860105cfe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [16, 16], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c880073c7680bd374402bf1d3bdcccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48a0d5d244daf2ad1cf50fe860105cfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_0c3e1d0576708afd33e6f8e7303f1888(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [8, 8], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_704ee27b080b101861d68d1165f3a8db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c3e1d0576708afd33e6f8e7303f1888
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_16f6618af7bf24eaa9ebe8c06b6e1af2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [4, 4], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_79a0f825cb7a2a5f74ffae260221caa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16f6618af7bf24eaa9ebe8c06b6e1af2
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_20419186b622176d134f855439d8b73e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a1eac9837492d6f3ed044744078bdbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20419186b622176d134f855439d8b73e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2c1814715664265b31c152d206652784(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9b27250c6db2ad69bc683f1c5d11ad48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_17226eb80019f6ecf60a2c7f24a8c899(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7239ed3503e05ea5a13d71ff0f923045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dc391419c29a7f57365b2c1c07c33960(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_39a1bec7d180170b4afa578e814f648c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 88, 88], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_635e2e7962950499329347389b646c4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_75b7239b9c9a1082e5a3beba4a4290c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20419186b622176d134f855439d8b73e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_41b3acbbb5a9b583eca540c14bc19fb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16f6618af7bf24eaa9ebe8c06b6e1af2
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1be7cf1f32d241d8452572c9f32a7dbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c3e1d0576708afd33e6f8e7303f1888
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d920f52a9889ca983202590df77cd0fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48a0d5d244daf2ad1cf50fe860105cfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_eb1e974cb94ffadc1804e9f74ba41ab4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6934dbc00876debe4bb3f1743e85c06a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_086bc3ddf275785e0d63c45284288e63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_390ff0135bed87c4bba8fbaf3f8cb1c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6334e5fc4aa9f314b26b2ee0d54efceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1536, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9b023be38017244485e4276b4d8cc2d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_36f3ac1de9c63479229ee79895bb3282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 2048, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b399e8ec61ce9109069c1fbf9dcf5c6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4447a1262049d0d2f5e9fb2dfb60664b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c204094c655f26dfaaeb653ad2ebc4de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c11f7f6ce6cbb4da135cffbcb3447186(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_41ad5b023cf7c85bdda2131bd85322f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 11, 11], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a9fc68fd7d21f9ae3ed275d1193329af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6cf205e000b4deaaf4f6bb71fe5f6a8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c6e23aa5de0a654a99ea03f116500ffa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 60, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a4ff797d8f35b9d7eb8a4e88d0a7d31b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_50197a01770a6d6b27085709b301d2bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_415edff1294f3ccacc35fc013d8d88aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c21ba27ba0454877fe9f72701b0d2afc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_596f89d33d8a9643351a7795bf42f713(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a93c5d2c81c5b9c77bb950c9cac0116a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 80, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_43cefcba941807eabbe1bd61681bac18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dc48a244f43c666cfc5e0a288f35f166(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dab4e61f6d041b8e6be0dd381b8e42d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_676db15017c02f33843aed4d8ddd20b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 88, 88], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1590cb8263d319d39a58f2778771d4f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 17, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f0bd56cf6f7ed0406361740c95a7b876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 17, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f44432eeab27579383e4fdb8cc62e3df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 17, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_494d6003fb5cb4fe0b3ca91f9a83287a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2f78d4260fd3f083ddd06e74a7df0a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_18dae5c3317bebaa560c5260ed1e1d06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_936c8107e69fc7ed6a1b0e6d431fd83f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_44de525f6829cdaef42e6f20e904e177(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cc9ed26c1c8ecfc23b7ab8aed7f59c13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 1, 49], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_84841cd35c7568edbfd81c222b9c70f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 15, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4ec8fd652f3da78abab3bade108a0c3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 15, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_209e0679bc0fa4bd4c8f72badcf5b357(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 15, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_330906194800216608b51dc127bd7fcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_869bd5272f5ec53ff3157d012ac09378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 16, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_77b1b7de8c8f65e24ca2f7a1db649a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_88084eef3dd85fff25f2e3207294d3ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 624, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_defb03792419f0b1aa9667942a0a4a93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_922f09b0d5fc0252425ce4679287950c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 44, 44], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0c61c451651174fd745732ddb25e9ccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 44, 44], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3a74044697cbae36de4480f6365c99b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([171, 60, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_51e71e99ebf636d51d8b8c06af7b093c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_49fd5af08deb3bbf9c126a811ae42f35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_aee0a97802d720afcc7702c6c5f5ec4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5a220151d6c57ad57e84d519fe09b338(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2321f3a16e3238069e0437ec1b0294c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7541e5249db781a565f44c04a9a6c02c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dc48a244f43c666cfc5e0a288f35f166(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7d2963610eb60926d20b37bcdfe8b11d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([145, 36, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e1dd299b22f569c145112b8f02711608(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1597aacba4826fbfce850a258a11eec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_eeceb4dc20bba1c9988ae480271580a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cbc28eb961a2d1e923c9d84cc05ef51d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20419186b622176d134f855439d8b73e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 38, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_78dd66ce985789fb62a29dea538fce98(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d525357bd80a9ce279f256e56fd062d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78dd66ce985789fb62a29dea538fce98
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1b5843326a31a515a138e1c3901f43d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5fc939949eec1115517a29f3d369e8c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1daf10bc6960c234a9a8b93d7169f24d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_245a8f6336ea3873d154a7c9610edfec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_640c99b06acba57f9592f58cee75ad5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8ab661e8dac69f6abcbbc0f8dde57403(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d470e30c97b6d75921b86502bf9d0fb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0d05f2521df2aa046a1e156b01484285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20419186b622176d134f855439d8b73e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_083295808f9c9614e1a69cf93ecd9099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16f6618af7bf24eaa9ebe8c06b6e1af2
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_19c3fafe88b8fa2c9a9aafcbe17d6bb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c3e1d0576708afd33e6f8e7303f1888
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b6b97cfda5aeda52bbec09c902229d89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48a0d5d244daf2ad1cf50fe860105cfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a595e78c868b290395af89ecd510851d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a28d6f0081e4c35854131314c6454096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_90d7c24831fd83eeaf3d5ca8bdea4e4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f8c10998192429c250c01939a1d8d128(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ea39489fb0d4dc48a3d1c98dc6765d98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d5e8eff6d4c8bfd6a14e3fe7200ff434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 68, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_af024879b555daefce1677d890741733(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 21, 21], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_28a6a380284b54f2518699e302bd42fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 21, 21], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9296caab94091bb479421943fb125e5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 21, 21], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_3f73e16a772cb40c9347b5c4a0e91c60(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [7, 7], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f6531d478f86d07a69b005eb06453abc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f73e16a772cb40c9347b5c4a0e91c60
    def get_inputs(self):
        return [
            paddle.uniform([11, 704, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_555dd4d9d48d56b9be85e0888ba567bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48a0d5d244daf2ad1cf50fe860105cfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_59326517412a36569454235127688b83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c3e1d0576708afd33e6f8e7303f1888
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_45a67363e1f64c22ed8fc5c3dfbcae0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16f6618af7bf24eaa9ebe8c06b6e1af2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2724d12da56cdd432ef91909ea19d3e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20419186b622176d134f855439d8b73e
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ae618051190ea87528ab5f347a27eb3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_54d5738a79682a7bcb408b65fde72c16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6ca82479e7b9b90566fcb449e21285e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6c7d03328f60a139c6ced99b803e1188(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([171, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_63b9008a4d2c98849ca9043f8dfb004e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 2048, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e10d606e34f35176a4b4de18843877e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a45134e6c3b0071feca4d51a2da4616c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_04dd5414ee5bb724db1659a51f4318e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_795b28eb1abae449fa89b796d0f32240(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c670f3948eb442654ccf1c6ab441a7fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_14e16bb1c7ac00c83247dcbfee103bc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d62e056341b2f71bd6aa96afa1e775cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 18, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9b023be38017244485e4276b4d8cc2d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8d5e35b7ec03786f716eaba131ae039c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ee1b90faf751255e356d2a7ff6485a74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6fd840f0d3a48beba21146ba5586fb0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 76, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c1d00fc49d354a0fb1e3b428ecef58ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 11, 11], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cd9a9abc08103098c18f71da87ecae55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 34, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_211fb06e7b557e1e589a43c4f38469e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f0c41774b6544ffcde79a79d83f057d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cfa079bf6392e598fae4f50676ca987f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0945f3d38a9e5cd841ef9a056654f2da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cd1a837f2f7e909ca9eae7069e5b5591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4d7326281dbb857bf5fca7977bda31e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_34a7ea90052f5a92020854efdf794837(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8cb6a64ccdda99d20957f7295e5f9652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 44, 44], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cd923b7f859ada92f12990ae99d1fac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 109, 109], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fb84581bc7fa4df6f2e6528304ce4ac0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c9e2cc5b43991bce2cfb65990ccdc4b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0f43ac5dc09211f23ca8dfdb5eab6631(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a1c10ee6959787d590f7d5531234921d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_32f7ccf5d716b7739d66ea46cc2ecc94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_becc4a98f9878d2e7f6d9328fd51d3f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5a2d6aa3faae3db19204dc959aa2f9d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_58412a459204944ab445c211baa29150(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f126cc0d5811af47887badcf73c04b37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f73e16a772cb40c9347b5c4a0e91c60
    def get_inputs(self):
        return [
            paddle.uniform([43, 704, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3c7e48296750d8ec5a52aa2d18cb8760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([145, 480, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_497c9078cb0136ef987e5ed0e7f1390e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_96e9105d8e78709f0252e4968150e4a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 320, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7855f4333c2989d079f99e946e17e3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2ca897c11baeacd0393947e1fda01952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_01f13d36cdba5a2be7bad4744b36b30a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2321f3a16e3238069e0437ec1b0294c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b363cc886ed0aec492365b5380155145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 36, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a958e48d08b6b14e5c3428654119a637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2a44004ea4868b68877f14a62ba7ecce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_15b1c06c1ea75f619ab0cf33894fef4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_430e78eedf015a61507ef4adfaf43c0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_601c0ae5d0c69a1896297ffe450ed91a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_69dd5fae6b9ba96326af22ce2024213c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_61b6d82f0a704c2d1ecc917fbada1ca4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c950a27d39f55f009e7dc7ebcae55f43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_87641573351dd44bb571485134ba2858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_245a8f6336ea3873d154a7c9610edfec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_640c99b06acba57f9592f58cee75ad5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8ab661e8dac69f6abcbbc0f8dde57403(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_384e57c8b5b27f3b0cc70ea5b663e3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c9dd2e619fe147b96532a52cbd3b9b17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c70ca5f097422048cdd831fe0d653e99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bc53e6eba979e3e1d17352d4b8798300(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 120, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4248a8edcc5c5052f0da7d6b9fb69614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 240, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1a75dcda8270060406a7c69db5f30cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_09dab437f1bbf989d1566322bb079be6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d1e16c6a2595cc09fbdc9f4c91a99ba7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 120, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e524c5e26993b120b916ea72c610e756(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 240, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c9009b37abd2d34dde6ad1fc87621c05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_080cb3750b8a3f53dca66aad9fd3d720(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 17, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5e3644df6995a0f25905c11f641c315a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_eeceb4dc20bba1c9988ae480271580a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_52a21a04c4fdf199c69a670e8983608a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_678716f0ca659fee71b8e07c4e05ccae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 44, 44], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a05b125e317301e597c56002bcff04f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1536, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e450d361c461abf738adb0197b390c70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7239ed3503e05ea5a13d71ff0f923045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bf650ff9f6689040d9cd589353bbe53c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_024f9f6e5b554345d614e942a31c6c25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 2048, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_37baf91e3400f0e962043244a91064b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e20ff2354595a752817e1a8b3a99359b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a1c10ee6959787d590f7d5531234921d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_666cbd1f7411803e44fbda6d62b972cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_666cbd1f7411803e44fbda6d62b972cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a5e3a924c19335848a383072a685f1c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a5e3a924c19335848a383072a685f1c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_76400c33fe7db4e02df91d52a87e910d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_76400c33fe7db4e02df91d52a87e910d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4190f08e13413aba9ceaa566354b2e6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4190f08e13413aba9ceaa566354b2e6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3f447d959755e5ef4dc150d05f4fde7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_12c567a16927b935b04e45abf6be7fc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d887f9e32c1fe25f0124583b3adbc945(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4a614f4180d62db034f0812b47a8ad91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f26b0a5e0dd0f5aa2483116efa4a900c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 76, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c74ac8cf69ef4e4fcc7c1d2409823434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dd71b64ff26b6786d81b8e289fa215f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cc7eac32fc1124d0ce16f926e5f92318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_19bb44d29acfa4fa11c3ec049d204a75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_43cefcba941807eabbe1bd61681bac18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_851f2382d0506a2fb97b4a3e1e555c39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 22, 22], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_62f8cccfd2663d34808900742febb8a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NHWC', 'avg', False, True, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6130e11fb9f92cf254147f13bb0aadf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62f8cccfd2663d34808900742febb8a8
    def get_inputs(self):
        return [
            paddle.uniform([22, 7, 7, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_596f89d33d8a9643351a7795bf42f713(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6327f1b678bb80ff938207b44ebcef52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dc73aec220909c02458b8be42c410f62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7fd98b312bd9d4a824176fb2de056ffc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_256dab955316dd45e1b2c726ed6a2afc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a23c2b8d99d39b5ce6e1be56be87f3ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bfc6046c487f6fdfd77734ac26592915(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_486bae1422ffa78732f0bc412ff2d078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9a34a2e751eb556cbb637f31a18f3d5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_69c42bf8984dbbce794d0a119997e33b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 22], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e3568d8f09f03796dcaa00458db6b90e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 1280, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8cf3d395d5493c58591a111f7e2e8136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8cf3d395d5493c58591a111f7e2e8136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_61b4f47c96e013de674b4b4ab0c1df23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_61b4f47c96e013de674b4b4ab0c1df23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bdb801b47f8ca3f7990a1240e993a7e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bdb801b47f8ca3f7990a1240e993a7e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1f4c7334bb70199d64cfcf61abb1a095(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1f4c7334bb70199d64cfcf61abb1a095(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e10d606e34f35176a4b4de18843877e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5aaeee4577bd0ba1b960ea5020544094(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 18, 27], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_58c05cf3476449a27476bb02b93c89b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 109, 109], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f20c89824f3819d7061415bf6f5e2331(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5edc70d71a8a1fd40b2397c4f3069931(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([11, 512, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_61e343925a98d0f8fff5413ce6e317b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9df0f5d9717c8097a97eef5060b93c7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 156, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e795391716780212bec5e6d9ae8bf134(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a416061db44630614861cf0369e00e8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 109, 109], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_01ee0426eb13df053c138e6e6ac5e4da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4cd20e539294abcc6c069f5fc21fadcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f4d71b018fa2fc534d797c14b18d930c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f3780326af56e1d11c40ce1b699c0f25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_07be2d3a1592de50721ca60d77d7104f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_07e993dc3cc7f15d90a6a440a657473b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3da5a699e4d24025c5bff4a8e2935ee1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5800eb9ce9b0c62324f1621a48ac49f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1a9aa5c797a5eb84a00446957b000595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3350b1cd48effdac0e43e2e344328e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 18, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_979a379c473b2cf02975ac73f593379c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 15, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e20ff2354595a752817e1a8b3a99359b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d85fe5aeb6b381237fea8f7e253d5f1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 23, 41], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_eac9008db2ea17fa9c6a475497164ade(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 46, 82], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0cc1a4b84c91f21b6bfaa047111a80a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 92, 164], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c6c34751857eb895b2896974dd275795(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 328], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_db86e133d6d86f70ecc57a583e55912b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 23, 41], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_84b4fdae96927a626a0520fa1310e002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 46, 82], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a231ab55c72ff46c724c4032998ca8a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 92, 164], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_58b0bd84c8a323c5d67e8854339b8048(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_52a21a04c4fdf199c69a670e8983608a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_581c6c87958b7f079aa34e0774a55968(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5ef19d8e22be218b6991ed3c55dfb81e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5ef19d8e22be218b6991ed3c55dfb81e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_363b2b4bea9cc8548fcd360ca8b625d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_363b2b4bea9cc8548fcd360ca8b625d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_747b31f55692b398e08fbe526fef2ea1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_747b31f55692b398e08fbe526fef2ea1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a0b279740c554b3e0f80d0aeeafd5006(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a0b279740c554b3e0f80d0aeeafd5006(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_72ef6ce4d9c247a53c81d658f8550f72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_71a56a3908afbd18dfd866bec4806a75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fa46e341d2155b87ad4ec7d12083a4d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78dd66ce985789fb62a29dea538fce98
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_64bb70fa86eacea4cd19ef5d0ea574ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9dc3a0dcfbcea9da49300a84d7e52220(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ccb03e3f03fc2c48006f2b99eefe034e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d555e5de984c950260e38da8f8e88e5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 34, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c25990f53e682a5ccf09003a5b27fda9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e674e42da654fa73394e6a3ac0f41993(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 480, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4da812c59aaf7229d3ee9897a9657b7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c354c3f973b23e0e68c1eae7537f4f08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 22, 22], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_13d1a7f455953409c6678e19a11a96f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9d7f61324cbd17addd5d95eaa3646f64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b5564d899771d006a698fc907e30a33e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_379452a23a216d688601b2abaabd17c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 23, 23], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_26261fe4264ef99bb227b7c04fdcf508(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 23, 23], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0e903f1b9520807319d3a13b160df2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 23, 23], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4e215bae2b44c5ea934e15ca13cb9c3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48a0d5d244daf2ad1cf50fe860105cfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_da9f3253e96702d826a5d6d0da73d628(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c3e1d0576708afd33e6f8e7303f1888
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_aa52be973f09717a3890a17df872893e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16f6618af7bf24eaa9ebe8c06b6e1af2
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1b7f95cdf133e1ac4c72315f2d2a0a11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20419186b622176d134f855439d8b73e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_94fb7102c7c2667f768935cff1eafa3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_19f6232f515aca4a14e4a1aa6b1fc061(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 1, 49], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_167bf02d711f2b17c48c3a00b3ed4301(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_07172e1cdc7d346be7bd9b146c8ae56a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_55de0816e6a420fbf263e6ca6463e7ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 17, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8249ff7d4bb087e4a8c8a8fbda9a7446(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 9, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_09827dd9c5886594c37cbc82ae0f6250(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 109, 109], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b0ce9ae50330cdab2fe1f70e13132e04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dbbef90d1859aef9c9e890578c027604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([43, 512, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ccca24f3b3b7d2c151c97ee289e2e539(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_74fa6cf9a670a96658861a8cabefecb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_777ef357e019c0b09882cd65b484b103(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_76d59de0f8710f48acedb5f112be69fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_33dc6753c5f403d9d45e5c11ed3846a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 92, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1fb8e3e749de388f304081422528db72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_80eef1b138407dbcdfbdb24aeae858be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b47bacb2e48bb7cca266feadb42d00ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_29d72d5c69f0579a72542273c762025a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_aee0a97802d720afcc7702c6c5f5ec4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_591e5952d31ec8f8ba10c055d3ae5be3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f708bc42f902a42b50f862308a13712a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([171, 36, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4b90f9e6c19b5489409b415d10a41a86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0fcb4dc583971b8eaae8b885a0500b15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8fe5239ffb5f0d8e7534f045326cce15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 1280, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_245a8f6336ea3873d154a7c9610edfec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_640c99b06acba57f9592f58cee75ad5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1d0c1b1db8a3acf45aab37cce81bcbfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_975ed18c5708a788fb9af91fcf862168(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_57c4a672842cea58c55f0867485d5ff9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a6a508ad76fe9485e81e3bb308713c9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0d1783748ef0e3e6ce2892b120692598(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e910d4d679769545dcf5257ec59b5347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([145, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3c18a386737115bd9742c2d275d064d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_30523f7a49d111c6ac26bf1c091828aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9b6e794a41ae03e33606a68864f67409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_46d0c8a90644511e6991a09d916d2c6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_851f2382d0506a2fb97b4a3e1e555c39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 22, 22], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9727ce7bc43535b774ee4b1426ca8fcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e9fdd1151f46d33ba7dd2d75458a2a92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_489dac4c312ab0268b1115b08674fe9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ccb03e3f03fc2c48006f2b99eefe034e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2d3609cdbac2e6492735e16182c0331f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b72c35d3f3eaf4e3f7fe219489832559(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_72ff878fd86337fee925be25aed207ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 480, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_32f7ccf5d716b7739d66ea46cc2ecc94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_07ad305873ce18ba368b846e61c6227e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 21, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_1f4b755751f58c1ffd3997526f0e286f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_369ba8844719b5ecc6c535a3701783eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f4b755751f58c1ffd3997526f0e286f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4613ee87f77aca1a1e4b47e802e5980e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f4b755751f58c1ffd3997526f0e286f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_54dc3476c568e5a8ff90790fd4023cc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f4b755751f58c1ffd3997526f0e286f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_39140fcd19043d3d9cbbbdd32786a970(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f4b755751f58c1ffd3997526f0e286f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_5371b647b256f531912bec9455c302b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [1, 1], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6830f00f4c64f8c1b54b89821b5e25b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5371b647b256f531912bec9455c302b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_97480e6ad7450433e4989da51f3a1630(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0fd702bc1d4d4310c02f89a9b47863c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_76ed74c0d90bbb5af94364ca5213a4e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f6b43abe5e9d962f7436353cb56b3128(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ab9552c8220f2b58ddb375a73ec3a444(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_da8b5637bf3dd146c272e5a1b14cb90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 18, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9f43304005eac2830de62e13c27e6f65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1024, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]




if __name__ == '__main__':
    unittest.main()