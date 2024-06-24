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



class PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cbd1bfd1aa21e231ff60929c1b6c31a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.573301076889038]]], [[[1.573761224746704]]], [[[1.7388144731521606]]], [[[1.749901294708252]]], [[[1.6451575756072998]]], [[[1.2924846410751343]]], [[[1.483615756034851]]], [[[1.3892743587493896]]], [[[1.75570547580719]]], [[[1.5007786750793457]]], [[[1.2848834991455078]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_2add8be0f7b7ef5b92b6ae5b2887495d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2add8be0f7b7ef5b92b6ae5b2887495d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b47207d47005ed2d331b2c4c07e75725(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_75664347dd983499db6fa1a0c53d1ab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([1841, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ca6406cd7fbc105bce4c62bf3254616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.1257723569869995]]], [[[0.9796621799468994]]], [[[1.0264205932617188]]], [[[0.9116493463516235]]], [[[1.7247947454452515]]], [[[1.543348789215088]]], [[[1.2471596002578735]]], [[[1.5492472648620605]]], [[[1.0709022283554077]]], [[[1.755483865737915]]], [[[0.9991201162338257]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_6b3cc149d530907745f9382fa3d6b64f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.294112205505371]]], [[[1.7862777709960938]]], [[[1.6645991802215576]]], [[[1.860257625579834]]], [[[1.2064704895019531]]], [[[1.0127906799316406]]], [[[1.3155906200408936]]], [[[1.3924102783203125]]], [[[1.1232943534851074]]], [[[1.7676327228546143]]], [[[1.8411645889282227]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_2add8be0f7b7ef5b92b6ae5b2887495d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9ab45c754506e025a3d725f35c121574(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b63ae5bbbf08697b75266b5b0b0b4d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ab45c754506e025a3d725f35c121574
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6de1dba226e53b061a6636626c97e6d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([5562, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2add8be0f7b7ef5b92b6ae5b2887495d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c7f55cd31bce525864df1e538c69a78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([1734, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5429e50f7e613bd7e89d0a3cd4b19eb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([1541, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_116bf27551deff634c6db9fe0de0f1c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0113849639892578]]], [[[1.710742712020874]]], [[[1.665868878364563]]], [[[1.5805926322937012]]], [[[1.1067363023757935]]], [[[1.6985406875610352]]], [[[1.4137792587280273]]], [[[1.8589251041412354]]], [[[1.2248902320861816]]], [[[1.0165685415267944]]], [[[1.4473178386688232]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_b1e9b4a8ed668a9fccb9bb436b94f7a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([2061, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17935ea693801345740ce728003613d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ab45c754506e025a3d725f35c121574
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_489a745eada28d648cdb0a1d3f556267(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bdedb8606495dc52593c27b2b5d2044f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_925f1ba8be857c1d091a694a056769a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ab45c754506e025a3d725f35c121574
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b8fdff533f2047ca6fe3aa5c4dd23591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([2369, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bad1479e2779b074c8848365e9440ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([3054, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa05f7797ed610ce1492d3736e6410ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([3819, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1237ec38a4635f586e986f09765b7446(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ab45c754506e025a3d725f35c121574
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56d8f05be5edefbc3f7931ef22550b20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3885812759399414]]], [[[1.0259177684783936]]], [[[1.5599579811096191]]], [[[1.8754490613937378]]], [[[1.5277831554412842]]], [[[0.951347291469574]]], [[[1.0226356983184814]]], [[[1.025580883026123]]], [[[1.427161455154419]]], [[[1.692711353302002]]], [[[1.921506643295288]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_2add8be0f7b7ef5b92b6ae5b2887495d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f86330a721db2abb058ebe4b46b13c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([2092, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_317efbfda08a2bab6adfcef3004ac5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ab45c754506e025a3d725f35c121574
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a24b8371cc357700f77b35ed6601ced(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([4214, 4], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()