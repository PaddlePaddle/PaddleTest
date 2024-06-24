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
        return False
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



class PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e83a7fa362a4fb3a8346e4b2af0faab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_550dfbfdedf7028244f37f529ea5c2ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3ce3791a0381899f3891b915c981372(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e2324d335b60f8c1c8f3f266088c48be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e83a7fa362a4fb3a8346e4b2af0faab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b14bef62bd9a54ce3216f6429a82823(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3ce3791a0381899f3891b915c981372(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45ce955bbcc9de9a9c98b92cc6240f4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b60e9cfc77aeab126e503754cccccaa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_653708b50e89657602cb6e6a523ae7d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e83a7fa362a4fb3a8346e4b2af0faab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1842d6b5159d9c03ccddf2a952df325d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb5ff2b0c52fa9a0eff18ec4bc1a647f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8690e0a80f8ecc3355ffa03e9203f8db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f2704db9329b01ae4c7eab3bf6f5f2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f3a64a4b7fc5554a8d37af935d4452d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4384515a55f9a8a52d52645ca6142bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b14bef62bd9a54ce3216f6429a82823(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3ce3791a0381899f3891b915c981372(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_267e4605f9db590a2dd35b1e2f6a1ca0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85497240b96ff723c3d6f5bf6c55569e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23fccefcc58c030395481ca53b316a7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6e051d08254367805aa89b8ebf7f63a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8690e0a80f8ecc3355ffa03e9203f8db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f2704db9329b01ae4c7eab3bf6f5f2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a29cedd8ad5ad6a339f931ba9948214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d5a029434f32c5f2aab1b0059482d21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ef9976719f6aa886d9a2a6540191a51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcc5ce465ef7e1225c96d3b299ef18e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da8791b7823227b1d2d0bf29706b3734(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23fccefcc58c030395481ca53b316a7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6e051d08254367805aa89b8ebf7f63a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad17e107bc1c8927376ff6c15551d965(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85497240b96ff723c3d6f5bf6c55569e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98fa16f8f388eb2a428647f86c9d30df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d59f00addeff2cf213d3659ab7e70a59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7c49d1b355da5c9e9311c98bc88f1d8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_735de30d62419e1331bdecb55e03b034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4384515a55f9a8a52d52645ca6142bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4384515a55f9a8a52d52645ca6142bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aec97dd99d6681a7502a9fbbbc1906a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a29cedd8ad5ad6a339f931ba9948214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad17e107bc1c8927376ff6c15551d965(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85497240b96ff723c3d6f5bf6c55569e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e83a7fa362a4fb3a8346e4b2af0faab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_550dfbfdedf7028244f37f529ea5c2ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3ce3791a0381899f3891b915c981372(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98fa16f8f388eb2a428647f86c9d30df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c7c3fe469f3dc615528ef38936b29dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f231928598ef69ec84cef22c1d350f73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4971ddc6f1e71dd5eca5bbc11e33c47e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6e051d08254367805aa89b8ebf7f63a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c7c3fe469f3dc615528ef38936b29dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f231928598ef69ec84cef22c1d350f73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4971ddc6f1e71dd5eca5bbc11e33c47e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6e051d08254367805aa89b8ebf7f63a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_735de30d62419e1331bdecb55e03b034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4384515a55f9a8a52d52645ca6142bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da8791b7823227b1d2d0bf29706b3734(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d59f00addeff2cf213d3659ab7e70a59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7c49d1b355da5c9e9311c98bc88f1d8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1842d6b5159d9c03ccddf2a952df325d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb5ff2b0c52fa9a0eff18ec4bc1a647f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d5a029434f32c5f2aab1b0059482d21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ef9976719f6aa886d9a2a6540191a51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2eca5c57b2425608f47570f689e1421a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98fa16f8f388eb2a428647f86c9d30df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0fbf1a23ed6d1629b40d6017d650b643(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98fa16f8f388eb2a428647f86c9d30df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_267e4605f9db590a2dd35b1e2f6a1ca0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85497240b96ff723c3d6f5bf6c55569e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45ce955bbcc9de9a9c98b92cc6240f4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b60e9cfc77aeab126e503754cccccaa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()