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



class PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c02aa2e7e2b8a69c9c715d0d88af463(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_933c7a15fc64e9acfe053bcf955238bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1be5ed938e550bd4524a9e26e34a6a18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_0066edc5450243aed008b5625d34c300(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c0758a718f143b161a9a07336bafc4e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0758a718f143b161a9a07336bafc4e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0758a718f143b161a9a07336bafc4e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0758a718f143b161a9a07336bafc4e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0758a718f143b161a9a07336bafc4e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0758a718f143b161a9a07336bafc4e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_35ba411914695f9db2eff8349ba6e42b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_948f140ff382e891d27384cc84c39672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_933c7a15fc64e9acfe053bcf955238bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1be5ed938e550bd4524a9e26e34a6a18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9904dedb93444ed01909dd4aed3c5ddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fde008691864bd0c9b02ea17523eb485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7cd5ca23d041f34cb3a04c84ae999225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7cd5ca23d041f34cb3a04c84ae999225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7cd5ca23d041f34cb3a04c84ae999225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7cd5ca23d041f34cb3a04c84ae999225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7cd5ca23d041f34cb3a04c84ae999225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7cd5ca23d041f34cb3a04c84ae999225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3ab9739e18f9d139eee242916da0d510(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_65c573334f29b71075b3fa1703e89d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ab9739e18f9d139eee242916da0d510
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3024, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e79466831a5069832d115c937da411a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7cd948b654fae4778a9f39e6771f8b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7cd948b654fae4778a9f39e6771f8b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7cd948b654fae4778a9f39e6771f8b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7cd948b654fae4778a9f39e6771f8b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7cd948b654fae4778a9f39e6771f8b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7cd948b654fae4778a9f39e6771f8b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eccc1a568a249b7ecc84b1d11c872657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ab9739e18f9d139eee242916da0d510
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4725, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_948f140ff382e891d27384cc84c39672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_42bcc75dd0cb7ae733bba72262a3280a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d5f0c2da62d087ee45117bc7a0ae1ddb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([171, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1be5ed938e550bd4524a9e26e34a6a18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4ab6a495cbb7de2bc9d842b996c9d977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_539024b98d34cfdd70fca13258ccb040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1f9e6111634aaa00dddd751aea59be10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f9e6111634aaa00dddd751aea59be10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f9e6111634aaa00dddd751aea59be10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f9e6111634aaa00dddd751aea59be10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f9e6111634aaa00dddd751aea59be10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f9e6111634aaa00dddd751aea59be10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_95aedbdc1a1638247b9dd6c49a16ad9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3969be97178a66cd76545c6a80bb4cfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_34cc3b1e97bc80c17724b680b4e03a9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06170189380645752], [-0.20056405663490295], [-0.30106890201568604], [0.25724780559539795]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.13348707556724548], [0.13511651754379272], [-0.333589106798172], [0.36184531450271606]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_9bf5f14bf5291ec5b7c7addf308ffa08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22580546140670776], [0.27939969301223755], [-0.3749677538871765], [-0.12680330872535706]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.12241360545158386], [-0.2222571074962616], [-0.29149335622787476], [-0.3536677360534668]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_f3fe963a9643ff41455f6921abc8847b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1605900228023529], [-0.37135812640190125], [-0.7974671125411987], [-0.20857638120651245]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_721aa0068835989109b1aa62009cf9aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.4928453266620636], [-0.5046300888061523], [-0.29695945978164673], [-0.734906792640686]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_5eab5806808239c343bea17e2465114d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.4950897991657257], [-0.4507054388523102], [0.4638780355453491], [0.4658241868019104]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.2940770983695984], [0.1707940697669983], [0.2271142601966858], [-0.3457677364349365]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_3c718729fc5f295d228ff17c79a5eb87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2870113253593445], [0.28237295150756836], [-0.1751812994480133], [0.38123905658721924]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.37043172121047974], [-0.32740920782089233], [-0.07800829410552979], [-0.2979702353477478]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_a9ee3dfda8ab7b59a3adbb0ecf9843e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ab9739e18f9d139eee242916da0d510
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.14321905374526978]], [[0.002419412136077881]], [[0.3457615375518799]], [[-0.48177680373191833]], [[-0.29202499985694885]], [[0.3379940390586853]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[4.135169982910156]], [[4.135169982910156]], [[4.135169982910156]], [[4.135169982910156]], [[4.135169982910156]], [[4.135169982910156]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_e0d651bd5d6f7b5a7d2fdded522a299e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37fa0a8929f5f8af842cc9080a03a0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4ab6a495cbb7de2bc9d842b996c9d977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4ab6a495cbb7de2bc9d842b996c9d977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_a9d8ec59a9524c49ea98c653ec112e69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_31a6e4f2bb998b95040535c1ec5d2c93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9d8ec59a9524c49ea98c653ec112e69
    def get_inputs(self):
        return [
            paddle.to_tensor([0.35411757230758667, 0.15644067525863647, 0.07635968923568726, -0.16245770454406738, 0.21228116750717163, 0.2637541890144348], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.16313570737838745, -0.4964006543159485, -0.4887825548648834, 0.09035730361938477, -0.25278910994529724, -0.2826343774795532], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_78d2ba1248c12d318d505ed12f616dee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9d8ec59a9524c49ea98c653ec112e69
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34914684295654297, 0.003970801830291748, 0.006298840045928955, 0.2024514079093933, -0.27971774339675903, 0.32279688119888306], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.005653113126754761, 0.05487024784088135, -0.15991780161857605, 0.29988062381744385, -0.2981359362602234, 0.12420547008514404], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_eb0f40dc97f2e4ad089e0a37c31deb98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9d8ec59a9524c49ea98c653ec112e69
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07483679056167603, 0.15644067525863647, 0.07635968923568726, -0.16245770454406738, 0.02825343608856201, 0.2637541890144348], dtype='float32').reshape([6]),
            paddle.to_tensor([0.17425143718719482, 0.4884592294692993, -0.3456558585166931, -0.042015135288238525, -0.11437886953353882, -0.039884716272354126], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_09661816f60c008aee9b34d20b17c1ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9d8ec59a9524c49ea98c653ec112e69
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.36654770374298096, -0.17453059554100037, 0.006298840045928955, -0.3943883776664734, -0.36228513717651367, 0.14998525381088257], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.15673527121543884, -0.45179250836372375, 0.42452186346054077, -0.2533150911331177, 0.2923128604888916, -0.4620075225830078], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_bd27ef872a0f60d8722fde12abe610ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_da854d1aa403749e0a6fcf76bfae50f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_793cc1a0a4888b2bf038d882faf86dd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5a342463cc06be833d9f884825701327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_933c7a15fc64e9acfe053bcf955238bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_da854d1aa403749e0a6fcf76bfae50f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bd27ef872a0f60d8722fde12abe610ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_948f140ff382e891d27384cc84c39672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_24b451db28f09b97a7c2a616d8d31c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fde008691864bd0c9b02ea17523eb485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4ab6a495cbb7de2bc9d842b996c9d977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9d2c14a0c817ab8f0d5b0ea920ff90da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6527b81865350bfebc6369132b32b9cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6527b81865350bfebc6369132b32b9cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6527b81865350bfebc6369132b32b9cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6527b81865350bfebc6369132b32b9cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6527b81865350bfebc6369132b32b9cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6527b81865350bfebc6369132b32b9cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4c9e9ef70781cf60c4d3bc9578cbb7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4c9e9ef70781cf60c4d3bc9578cbb7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4c9e9ef70781cf60c4d3bc9578cbb7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4c9e9ef70781cf60c4d3bc9578cbb7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4c9e9ef70781cf60c4d3bc9578cbb7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4c9e9ef70781cf60c4d3bc9578cbb7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c02aa2e7e2b8a69c9c715d0d88af463(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_455d79c4f1f4cef49174f93acd63b32f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([22, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_95aedbdc1a1638247b9dd6c49a16ad9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37fa0a8929f5f8af842cc9080a03a0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4ab6a495cbb7de2bc9d842b996c9d977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1be5ed938e550bd4524a9e26e34a6a18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7a529cbee4a7156c86282281b675a9c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7a529cbee4a7156c86282281b675a9c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7a529cbee4a7156c86282281b675a9c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7a529cbee4a7156c86282281b675a9c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7a529cbee4a7156c86282281b675a9c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7a529cbee4a7156c86282281b675a9c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_539024b98d34cfdd70fca13258ccb040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ace83efb6baa47caaaac89599882a3fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.24296218156814575], [-0.3867464065551758], [0.21708428859710693], [0.26299989223480225], [-0.16711819171905518], [-0.1711117923259735]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.002047300338745117], [-0.35312914848327637], [-0.2366151511669159], [-0.35074514150619507], [-0.015176147222518921], [0.3708425760269165]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_4879d0b04e834018b0e33d37c944ec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0286027193069458], [0.2744826078414917], [0.33321893215179443], [-0.018983185291290283], [-0.25323376059532166], [-0.10177692770957947]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.46462422609329224], [-0.13661116361618042], [0.3498750329017639], [-0.3446376919746399], [0.2325107455253601], [-0.26569807529449463]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_f74a93703046e175f30d178bfbcb1235(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.13190007209777832], [-0.3981360197067261], [-0.4627215564250946], [-0.05427950620651245], [-0.45611828565597534], [-0.48559895157814026]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_21aaf07f9815b4c7a3f775125ebe5b6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.13453972339630127], [-0.2907872200012207], [0.13322335481643677], [-0.7568634152412415], [-0.6738439798355103], [-0.15998604893684387]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_b7c7087d14c379b025de7ac1949c89f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05035901069641113], [0.011389613151550293], [0.09646791219711304], [-0.4840776026248932], [0.28900009393692017], [0.2621924877166748]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.13394737243652344], [-0.37528514862060547], [0.2261064052581787], [-0.2964656352996826], [-0.3431260585784912], [0.31448715925216675]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_50611f208849df772fd5f45cd57f1755(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.33008450269699097], [0.15417605638504028], [0.19999557733535767], [0.41222572326660156], [-0.3463072180747986], [-0.10571202635765076]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.3742426037788391], [-0.2767235040664673], [-0.47726520895957947], [-0.06877976655960083], [0.4206101894378662], [-0.22372296452522278]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_35ba411914695f9db2eff8349ba6e42b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37fa0a8929f5f8af842cc9080a03a0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_95aedbdc1a1638247b9dd6c49a16ad9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4ab6a495cbb7de2bc9d842b996c9d977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f205dd87b3fbfcc5a970eb785ae49f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_948f140ff382e891d27384cc84c39672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d00bd4aed24c9e3a2fbad1841c8949c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37fa0a8929f5f8af842cc9080a03a0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_539024b98d34cfdd70fca13258ccb040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d7dbc942533a350e71f544d7683d7da6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_35ba411914695f9db2eff8349ba6e42b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fed0d16f6174f6945c1018652b7ae100(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_24b451db28f09b97a7c2a616d8d31c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_539024b98d34cfdd70fca13258ccb040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a110ecbbd601844bd63cf2dc50f317a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_98ce31b9e45b5b463c34fe174aaa554a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.47831395268440247]], [[0.6133474707603455]], [[0.5916699171066284]], [[0.4318757653236389]], [[0.4837804436683655]], [[0.45515042543411255]], [[0.47219884395599365]], [[0.3662322163581848]], [[0.530521810054779]], [[0.463400661945343]], [[0.5923812985420227]], [[0.5909186005592346]], [[0.5779500007629395]], [[0.3995029330253601]], [[0.38871750235557556]], [[0.6150435209274292]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_35ba411914695f9db2eff8349ba6e42b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_24b451db28f09b97a7c2a616d8d31c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4ab6a495cbb7de2bc9d842b996c9d977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f205dd87b3fbfcc5a970eb785ae49f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8e25267beac0bd41c1dd36ba6dfa12b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_95aedbdc1a1638247b9dd6c49a16ad9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4ab6a495cbb7de2bc9d842b996c9d977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37fa0a8929f5f8af842cc9080a03a0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0c9fa7cbf29877e7f7154a2028387bd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c9fa7cbf29877e7f7154a2028387bd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c9fa7cbf29877e7f7154a2028387bd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c9fa7cbf29877e7f7154a2028387bd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c9fa7cbf29877e7f7154a2028387bd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c9fa7cbf29877e7f7154a2028387bd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_24b451db28f09b97a7c2a616d8d31c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37fa0a8929f5f8af842cc9080a03a0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_42bcc75dd0cb7ae733bba72262a3280a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_22312a913dd15a700710b469726122bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 960], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0bfe6114a2598fde2b8e865e1947187a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0bfe6114a2598fde2b8e865e1947187a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0bfe6114a2598fde2b8e865e1947187a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0bfe6114a2598fde2b8e865e1947187a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0bfe6114a2598fde2b8e865e1947187a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0bfe6114a2598fde2b8e865e1947187a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_793cc1a0a4888b2bf038d882faf86dd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_793135c7bd8d0692f23768b62ab9a06d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_793135c7bd8d0692f23768b62ab9a06d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_793135c7bd8d0692f23768b62ab9a06d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_793135c7bd8d0692f23768b62ab9a06d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_793135c7bd8d0692f23768b62ab9a06d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_793135c7bd8d0692f23768b62ab9a06d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b3890df94c367be68411dd285b805fbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ab9739e18f9d139eee242916da0d510
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8eb9ec3a2e0e50a0221e15ad310cd94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8eb9ec3a2e0e50a0221e15ad310cd94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8eb9ec3a2e0e50a0221e15ad310cd94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8eb9ec3a2e0e50a0221e15ad310cd94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8eb9ec3a2e0e50a0221e15ad310cd94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8eb9ec3a2e0e50a0221e15ad310cd94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2ba987d61fc6ab9eaf37d84881f1b949(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 624], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 624], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_948f140ff382e891d27384cc84c39672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37a3d5fbc8bfb5379fda2f24fc436db2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_42bcc75dd0cb7ae733bba72262a3280a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9d3963719a499f1069a1bdb8b84fd74e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ec836b3b57336cb068f9a730ec9cae66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8e25267beac0bd41c1dd36ba6dfa12b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_11adc58d21c532fec66b197210b79f41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d02489bd41c9438c9b268b804baf80a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d02489bd41c9438c9b268b804baf80a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d02489bd41c9438c9b268b804baf80a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d02489bd41c9438c9b268b804baf80a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d02489bd41c9438c9b268b804baf80a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d02489bd41c9438c9b268b804baf80a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_933c7a15fc64e9acfe053bcf955238bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_448b033d30e2f5afe2d00aa7ebc5c362(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_37fa0a8929f5f8af842cc9080a03a0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3abfce84cf4aa1c21e0ed987883c9c9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([145, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc420555eb62fef51c17c1d1adfc99be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc420555eb62fef51c17c1d1adfc99be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc420555eb62fef51c17c1d1adfc99be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc420555eb62fef51c17c1d1adfc99be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc420555eb62fef51c17c1d1adfc99be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc420555eb62fef51c17c1d1adfc99be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_11adc58d21c532fec66b197210b79f41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_37fa0a8929f5f8af842cc9080a03a0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4d28faec9221cd529a92f0c13c00504a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.17065510153770447], [-0.41795647144317627], [0.4431740641593933], [0.034448087215423584], [-0.10538581013679504]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.2499765157699585], [0.47538548707962036], [-0.3144863247871399], [-0.2073010504245758], [0.29379701614379883]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_d64cbb9ae04cece2cca19775b9dc04fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07766205072402954], [-0.4961780607700348], [-0.35865235328674316], [-0.21105095744132996], [-0.42650023102760315]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.22014576196670532], [-0.33925819396972656], [0.43902361392974854], [0.4746719002723694], [0.4194345474243164]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_8417237d00887252f0ed8b21c442e03c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16268768906593323], [-0.5031480193138123], [-0.6940600872039795], [0.08316034078598022], [-0.5448017120361328]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_189b9e91423324a2cc35d9a732199599(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.6636602282524109], [-0.9441380500793457], [-0.7952777743339539], [-0.6186395883560181], [-0.9083819389343262]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_8533d9a640abc8e42b00209a78b0a74d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.3333427906036377], [0.08519154787063599], [-0.3535339832305908], [-0.29046139121055603], [-0.3666517734527588]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.3873671591281891], [-0.13544422388076782], [0.3795737624168396], [-0.29945799708366394], [0.43941593170166016]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_7e0709cb4192d06d4d26b94126e0b744(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.44351446628570557], [0.06138598918914795], [-0.4839599132537842], [0.4075886011123657], [0.4818817377090454]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.26713675260543823], [0.4479599595069885], [0.4366254210472107], [0.08791720867156982], [0.42962461709976196]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_1be5ed938e550bd4524a9e26e34a6a18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5ffe864ba5aa83e1e270d8bf21070090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ffe864ba5aa83e1e270d8bf21070090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ffe864ba5aa83e1e270d8bf21070090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ffe864ba5aa83e1e270d8bf21070090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ffe864ba5aa83e1e270d8bf21070090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ffe864ba5aa83e1e270d8bf21070090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_508e8a79219f8fbc56b4a6eef7c899ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ab9739e18f9d139eee242916da0d510
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_539024b98d34cfdd70fca13258ccb040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_948f140ff382e891d27384cc84c39672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bda91cb9198a48b222051c40840fab91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4ab6a495cbb7de2bc9d842b996c9d977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37fa0a8929f5f8af842cc9080a03a0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_24b451db28f09b97a7c2a616d8d31c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d55cd8f79466237bd71ff2cf07b6de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2d55cd8f79466237bd71ff2cf07b6de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2d55cd8f79466237bd71ff2cf07b6de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2d55cd8f79466237bd71ff2cf07b6de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2d55cd8f79466237bd71ff2cf07b6de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2d55cd8f79466237bd71ff2cf07b6de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3df01832f08907482b1135e0973e789f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ab9739e18f9d139eee242916da0d510
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6069, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_95aedbdc1a1638247b9dd6c49a16ad9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_17241da29bb93890a18b000c8fac3948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_17241da29bb93890a18b000c8fac3948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_17241da29bb93890a18b000c8fac3948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_17241da29bb93890a18b000c8fac3948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_17241da29bb93890a18b000c8fac3948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_17241da29bb93890a18b000c8fac3948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cc68e201f245a36e8ddc174cce925bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([171, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_539024b98d34cfdd70fca13258ccb040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cf32fa6afa38a7bd357a13187d6a2ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7ec722b3d31e9c2eef30727defe23d81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37fa0a8929f5f8af842cc9080a03a0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_da854d1aa403749e0a6fcf76bfae50f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1d223f68b11c0546c41b476363839673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.012090981006622314], [0.014628350734710693], [-0.21090012788772583], [0.04272180795669556], [-0.05056363344192505], [-0.18745583295822144], [0.37026363611221313], [-0.12386777997016907], [0.04815411567687988]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.21110379695892334], [-0.381933331489563], [-0.16953760385513306], [0.35887253284454346], [-0.008764803409576416], [0.3931576609611511], [-0.009689152240753174], [0.019851267337799072], [-0.32272660732269287]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_0fdba3087f6505f72d9be09a5137a6fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.1128949522972107], [0.16002947092056274], [-0.2516271471977234], [-0.4568890631198883], [-0.3312219977378845], [0.040363609790802], [0.17029047012329102], [0.34896695613861084], [-0.12434554100036621]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.3303430676460266], [-0.18069246411323547], [-0.22718149423599243], [-0.31739091873168945], [0.11852103471755981], [-0.29862701892852783], [-0.12920188903808594], [-0.37988555431365967], [0.2331845760345459]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_f173773a5567e7f6ca61749b2fdc5a3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07215607166290283], [-0.2269076704978943], [-0.6258493661880493], [0.019590437412261963], [-0.2745057940483093], [-0.6176184415817261], [-0.3291023373603821], [0.05962526798248291], [-0.683196485042572]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_7212cf9094529fa32c9a98310e2d1a2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.3334689140319824], [-0.6014107465744019], [-0.7445532083511353], [-0.7998079061508179], [-0.3763255476951599], [-0.6437492966651917], [-0.21342188119888306], [-0.5940254926681519], [0.08150973916053772]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_b394d76530d56b82b48535a6edad1f2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.403987318277359], [-0.2360001504421234], [0.4149492383003235], [-0.014448881149291992], [0.22394216060638428], [-0.11358064413070679], [-0.2474491000175476], [-0.3083335757255554], [0.18501579761505127]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.08424705266952515], [-0.1550256609916687], [-0.4499405324459076], [0.023131370544433594], [-0.21213120222091675], [0.43016260862350464], [0.3194131851196289], [-0.18349304795265198], [0.36046987771987915]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_677b527678975c48cde20a007cdbd37c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.37472984194755554], [0.420718252658844], [0.49292606115341187], [-0.42530685663223267], [0.04510354995727539], [-0.457436203956604], [-0.46275776624679565], [0.16821861267089844], [-0.2780924439430237]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0031258463859558105], [0.2167760729789734], [-0.030280649662017822], [0.34291887283325195], [0.011862993240356445], [0.3451222777366638], [0.08421999216079712], [0.2141399383544922], [-0.20585528016090393]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_37a3d5fbc8bfb5379fda2f24fc436db2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7ec722b3d31e9c2eef30727defe23d81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_24b451db28f09b97a7c2a616d8d31c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5a342463cc06be833d9f884825701327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_da854d1aa403749e0a6fcf76bfae50f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c6953ef71fb2b526d6e5834b5b3e3fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ab9739e18f9d139eee242916da0d510
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.03960481286048889]], [[-0.033424556255340576]], [[-0.10159653425216675]], [[0.26393401622772217]], [[-0.4051783084869385]], [[-0.038964927196502686]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[4.135169982910156]], [[4.135169982910156]], [[4.135169982910156]], [[4.135169982910156]], [[4.135169982910156]], [[4.135169982910156]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_948f140ff382e891d27384cc84c39672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f205dd87b3fbfcc5a970eb785ae49f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_933c7a15fc64e9acfe053bcf955238bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8e25267beac0bd41c1dd36ba6dfa12b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b744adae652ccd3c0818f1d1286b3e38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b744adae652ccd3c0818f1d1286b3e38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b744adae652ccd3c0818f1d1286b3e38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b744adae652ccd3c0818f1d1286b3e38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b744adae652ccd3c0818f1d1286b3e38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b744adae652ccd3c0818f1d1286b3e38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b3890df94c367be68411dd285b805fbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ab9739e18f9d139eee242916da0d510
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9d3963719a499f1069a1bdb8b84fd74e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8e25267beac0bd41c1dd36ba6dfa12b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d02489bd41c9438c9b268b804baf80a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d02489bd41c9438c9b268b804baf80a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d02489bd41c9438c9b268b804baf80a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d02489bd41c9438c9b268b804baf80a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d02489bd41c9438c9b268b804baf80a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d02489bd41c9438c9b268b804baf80a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_85895d42e3d27f718278d29fcfad88e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7a529cbee4a7156c86282281b675a9c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7a529cbee4a7156c86282281b675a9c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7a529cbee4a7156c86282281b675a9c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7a529cbee4a7156c86282281b675a9c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7a529cbee4a7156c86282281b675a9c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7a529cbee4a7156c86282281b675a9c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_933c7a15fc64e9acfe053bcf955238bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6ce5a9955049e99ede4a4a1ddba1dfe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ce5a9955049e99ede4a4a1ddba1dfe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ce5a9955049e99ede4a4a1ddba1dfe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ce5a9955049e99ede4a4a1ddba1dfe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ce5a9955049e99ede4a4a1ddba1dfe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ce5a9955049e99ede4a4a1ddba1dfe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_539024b98d34cfdd70fca13258ccb040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_da854d1aa403749e0a6fcf76bfae50f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2b3cd063abfccf02a1d14c41a641973a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b3cd063abfccf02a1d14c41a641973a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b3cd063abfccf02a1d14c41a641973a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b3cd063abfccf02a1d14c41a641973a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b3cd063abfccf02a1d14c41a641973a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b3cd063abfccf02a1d14c41a641973a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10e56b396706c4103ce0029e0386b174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([145, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_37fa0a8929f5f8af842cc9080a03a0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_da854d1aa403749e0a6fcf76bfae50f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_85895d42e3d27f718278d29fcfad88e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_35ba411914695f9db2eff8349ba6e42b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f205dd87b3fbfcc5a970eb785ae49f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3e561568560785cb4bc44be926dd2649(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f205dd87b3fbfcc5a970eb785ae49f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4c02aa2e7e2b8a69c9c715d0d88af463(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_24b451db28f09b97a7c2a616d8d31c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f205dd87b3fbfcc5a970eb785ae49f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0de0de9aba88b98edee304a196fb5b1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0de0de9aba88b98edee304a196fb5b1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0de0de9aba88b98edee304a196fb5b1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0de0de9aba88b98edee304a196fb5b1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9268df7fce69f8d626368b14a24d54e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9268df7fce69f8d626368b14a24d54e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9268df7fce69f8d626368b14a24d54e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9268df7fce69f8d626368b14a24d54e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_85895d42e3d27f718278d29fcfad88e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5a342463cc06be833d9f884825701327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_35ba411914695f9db2eff8349ba6e42b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_24b451db28f09b97a7c2a616d8d31c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_539024b98d34cfdd70fca13258ccb040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_24b451db28f09b97a7c2a616d8d31c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_862f493f0e5017105538410042627cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_862f493f0e5017105538410042627cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_862f493f0e5017105538410042627cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_862f493f0e5017105538410042627cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_862f493f0e5017105538410042627cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_862f493f0e5017105538410042627cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0263a54bbca818825b4ba6c19cc3f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0263a54bbca818825b4ba6c19cc3f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0263a54bbca818825b4ba6c19cc3f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0263a54bbca818825b4ba6c19cc3f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0263a54bbca818825b4ba6c19cc3f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0263a54bbca818825b4ba6c19cc3f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4ab6a495cbb7de2bc9d842b996c9d977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_948f140ff382e891d27384cc84c39672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_055fe8fefd401fa3a39053036a511ef3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0361386239528656]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.24892032146453857]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_7fb7b924372de7839dc34cbcda571741(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2491803765296936]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.4878164529800415]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_db2468239159ecd9e7b84cf6c9e9cb32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.37545904517173767]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[3.402820018375656e+38]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_03d2335a18100226842b594b21334989(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.7754269242286682]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[3.402820018375656e+38]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_ad808130fae1eef04ce620aa998537b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33932042121887207]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.05973100662231445]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_cd5fbc848ecb3bdf45ca7edccfc18632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1952732801437378]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.2876104712486267]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_948f140ff382e891d27384cc84c39672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_da854d1aa403749e0a6fcf76bfae50f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c0263a54bbca818825b4ba6c19cc3f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0263a54bbca818825b4ba6c19cc3f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0263a54bbca818825b4ba6c19cc3f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0263a54bbca818825b4ba6c19cc3f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0263a54bbca818825b4ba6c19cc3f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0263a54bbca818825b4ba6c19cc3f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_933c7a15fc64e9acfe053bcf955238bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8eb9ec3a2e0e50a0221e15ad310cd94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8eb9ec3a2e0e50a0221e15ad310cd94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8eb9ec3a2e0e50a0221e15ad310cd94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8eb9ec3a2e0e50a0221e15ad310cd94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8eb9ec3a2e0e50a0221e15ad310cd94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8eb9ec3a2e0e50a0221e15ad310cd94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_85895d42e3d27f718278d29fcfad88e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_95aedbdc1a1638247b9dd6c49a16ad9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_24b451db28f09b97a7c2a616d8d31c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_da854d1aa403749e0a6fcf76bfae50f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c0758a718f143b161a9a07336bafc4e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0758a718f143b161a9a07336bafc4e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0758a718f143b161a9a07336bafc4e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0758a718f143b161a9a07336bafc4e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0758a718f143b161a9a07336bafc4e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0758a718f143b161a9a07336bafc4e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_332dfc27f1c9d5807a62d72b3a20e3c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_332dfc27f1c9d5807a62d72b3a20e3c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_332dfc27f1c9d5807a62d72b3a20e3c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_332dfc27f1c9d5807a62d72b3a20e3c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_332dfc27f1c9d5807a62d72b3a20e3c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_332dfc27f1c9d5807a62d72b3a20e3c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2dd4c9d0cd57499b886cbc9a43ccdcfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ab9739e18f9d139eee242916da0d510
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 11109, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4ab6a495cbb7de2bc9d842b996c9d977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f205dd87b3fbfcc5a970eb785ae49f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0bfe6114a2598fde2b8e865e1947187a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0bfe6114a2598fde2b8e865e1947187a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0bfe6114a2598fde2b8e865e1947187a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0bfe6114a2598fde2b8e865e1947187a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0bfe6114a2598fde2b8e865e1947187a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0bfe6114a2598fde2b8e865e1947187a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f205dd87b3fbfcc5a970eb785ae49f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_35ba411914695f9db2eff8349ba6e42b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_862f493f0e5017105538410042627cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_862f493f0e5017105538410042627cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_862f493f0e5017105538410042627cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_862f493f0e5017105538410042627cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_862f493f0e5017105538410042627cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_862f493f0e5017105538410042627cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8ea1dcbe38c6e6c48f1208ff852bbf65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8ea1dcbe38c6e6c48f1208ff852bbf65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8ea1dcbe38c6e6c48f1208ff852bbf65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8ea1dcbe38c6e6c48f1208ff852bbf65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8ea1dcbe38c6e6c48f1208ff852bbf65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8ea1dcbe38c6e6c48f1208ff852bbf65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c39132240900f1811c3c74e2e240488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ab9739e18f9d139eee242916da0d510
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2100, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7ec722b3d31e9c2eef30727defe23d81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d00bd4aed24c9e3a2fbad1841c8949c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_948f140ff382e891d27384cc84c39672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c0c27082ffef3e294272d71b44669215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0c27082ffef3e294272d71b44669215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0c27082ffef3e294272d71b44669215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0c27082ffef3e294272d71b44669215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0c27082ffef3e294272d71b44669215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0c27082ffef3e294272d71b44669215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_508e8a79219f8fbc56b4a6eef7c899ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ab9739e18f9d139eee242916da0d510
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f205dd87b3fbfcc5a970eb785ae49f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1be5ed938e550bd4524a9e26e34a6a18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_24b451db28f09b97a7c2a616d8d31c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1f9e6111634aaa00dddd751aea59be10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f9e6111634aaa00dddd751aea59be10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f9e6111634aaa00dddd751aea59be10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f9e6111634aaa00dddd751aea59be10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f9e6111634aaa00dddd751aea59be10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f9e6111634aaa00dddd751aea59be10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_42bcc75dd0cb7ae733bba72262a3280a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_da854d1aa403749e0a6fcf76bfae50f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_da854d1aa403749e0a6fcf76bfae50f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e79466831a5069832d115c937da411a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6527b81865350bfebc6369132b32b9cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6527b81865350bfebc6369132b32b9cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6527b81865350bfebc6369132b32b9cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6527b81865350bfebc6369132b32b9cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6527b81865350bfebc6369132b32b9cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6527b81865350bfebc6369132b32b9cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3958f288e744fc925683a2c45c51503a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 156], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8e25267beac0bd41c1dd36ba6dfa12b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_df1aa744d67fc27554b7f9b30c15eebc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.47120556235313416]], [[0.46167635917663574]], [[0.46761423349380493]], [[0.42727261781692505]], [[0.42172589898109436]], [[0.46778908371925354]], [[0.3988942801952362]], [[0.6166152954101562]], [[0.5590395927429199]], [[0.5651613473892212]], [[0.42951393127441406]], [[0.5848869681358337]], [[0.46537312865257263]], [[0.5061679482460022]], [[0.4675232470035553]], [[0.4999001920223236]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_af08d0d2c07225ae8596c30ed034501c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9d2c14a0c817ab8f0d5b0ea920ff90da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_539024b98d34cfdd70fca13258ccb040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_539024b98d34cfdd70fca13258ccb040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9d3963719a499f1069a1bdb8b84fd74e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bd27ef872a0f60d8722fde12abe610ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37fa0a8929f5f8af842cc9080a03a0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37fa0a8929f5f8af842cc9080a03a0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37fa0a8929f5f8af842cc9080a03a0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37fa0a8929f5f8af842cc9080a03a0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c9b892e524bc128c4b30dadfd754ecaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.5654497146606445]], [[-2.9553353786468506]], [[0.6897178888320923]], [[-2.8710885047912598]], [[1.9228309392929077]], [[-1.0404026508331299]], [[2.472787857055664]], [[3.839804172515869]], [[2.959489107131958]], [[2.2106380462646484]], [[2.7227871417999268]], [[3.9583899974823]], [[-3.6626930236816406]], [[3.0224733352661133]], [[-1.8645009994506836]], [[-0.09688568115234375]], [[0.9109943509101868]], [[0.8171284198760986]], [[1.4595714807510376]], [[-0.06313318014144897]], [[2.156186580657959]], [[2.6706206798553467]], [[4.171671390533447]], [[2.082115650177002]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_69732fa50427219b388df8a1b209eb7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.151217937469482]], [[4.9136271476745605]], [[-0.39605921506881714]], [[4.811964511871338]], [[4.512101650238037]], [[0.03443711996078491]], [[-1.6091830730438232]], [[3.1132774353027344]], [[-2.0338170528411865]], [[-0.8647464513778687]], [[1.5548102855682373]], [[0.5183814764022827]], [[-1.2889149188995361]], [[7.342719554901123]], [[-1.5875318050384521]], [[-1.686166524887085]], [[-2.4593350887298584]], [[-3.941873073577881]], [[2.5025947093963623]], [[1.1122729778289795]], [[2.9859533309936523]], [[-0.2216755747795105]], [[0.18954044580459595]], [[-0.49860239028930664]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cf8fbb2ef0bbafa399abb9bf65d1ab24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.2280588150024414]], [[-2.0571506023406982]], [[2.533856153488159]], [[-2.1879022121429443]], [[4.369203567504883]], [[-2.833670139312744]], [[-3.63808012008667]], [[-3.6585612297058105]], [[1.821708083152771]], [[1.5540488958358765]], [[-3.822190761566162]], [[3.1135730743408203]], [[-0.49642717838287354]], [[-0.570594072341919]], [[-0.8257262706756592]], [[-0.1906757354736328]], [[-1.6478404998779297]], [[-2.4280030727386475]], [[-5.167953968048096]], [[2.6113617420196533]], [[-0.45281535387039185]], [[-0.9053754806518555]], [[2.5657553672790527]], [[-0.6423838138580322]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_117b4f3d7e2ef1cfd3ceaa98471d0405(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.17186620831489563]], [[0.5234401822090149]], [[0.3560965359210968]], [[0.4454316198825836]], [[0.4945691227912903]], [[0.339041531085968]], [[0.0944085419178009]], [[0.3754713535308838]], [[0.7612360715866089]], [[0.4362524151802063]], [[0.3742596507072449]], [[0.3421916961669922]], [[0.491133451461792]], [[0.7876287698745728]], [[0.6150720715522766]], [[0.5835603475570679]], [[0.5477710962295532]], [[0.34450241923332214]], [[0.36460086703300476]], [[0.5777625441551208]], [[0.4160816967487335]], [[0.6061511635780334]], [[0.5422899127006531]], [[0.6924923062324524]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bd27ef872a0f60d8722fde12abe610ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_948f140ff382e891d27384cc84c39672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5a342463cc06be833d9f884825701327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_933c7a15fc64e9acfe053bcf955238bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37fa0a8929f5f8af842cc9080a03a0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4ab6a495cbb7de2bc9d842b996c9d977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a4c9e9ef70781cf60c4d3bc9578cbb7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4c9e9ef70781cf60c4d3bc9578cbb7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4c9e9ef70781cf60c4d3bc9578cbb7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4c9e9ef70781cf60c4d3bc9578cbb7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4c9e9ef70781cf60c4d3bc9578cbb7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4c9e9ef70781cf60c4d3bc9578cbb7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_95aedbdc1a1638247b9dd6c49a16ad9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37fa0a8929f5f8af842cc9080a03a0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_948f140ff382e891d27384cc84c39672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37fa0a8929f5f8af842cc9080a03a0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_57d6e7c9065e5bd99de09759bc020b00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([22, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c4864dfc1733d3943ff36f11952f67b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 872], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fde008691864bd0c9b02ea17523eb485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_17241da29bb93890a18b000c8fac3948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_17241da29bb93890a18b000c8fac3948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_17241da29bb93890a18b000c8fac3948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_17241da29bb93890a18b000c8fac3948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_17241da29bb93890a18b000c8fac3948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_17241da29bb93890a18b000c8fac3948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e0d651bd5d6f7b5a7d2fdded522a299e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7ec722b3d31e9c2eef30727defe23d81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4c02aa2e7e2b8a69c9c715d0d88af463(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37fa0a8929f5f8af842cc9080a03a0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0c9fa7cbf29877e7f7154a2028387bd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c9fa7cbf29877e7f7154a2028387bd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c9fa7cbf29877e7f7154a2028387bd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c9fa7cbf29877e7f7154a2028387bd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c9fa7cbf29877e7f7154a2028387bd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c9fa7cbf29877e7f7154a2028387bd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c4864dfc1733d3943ff36f11952f67b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 872], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_381b6370ca15605ca670df80a2ee09e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_381b6370ca15605ca670df80a2ee09e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_381b6370ca15605ca670df80a2ee09e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_381b6370ca15605ca670df80a2ee09e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_381b6370ca15605ca670df80a2ee09e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_381b6370ca15605ca670df80a2ee09e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a5d6682fb78f69b43829b42a22750d60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ab9739e18f9d139eee242916da0d510
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8400, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1be5ed938e550bd4524a9e26e34a6a18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_da854d1aa403749e0a6fcf76bfae50f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37fa0a8929f5f8af842cc9080a03a0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6ce5a9955049e99ede4a4a1ddba1dfe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ce5a9955049e99ede4a4a1ddba1dfe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ce5a9955049e99ede4a4a1ddba1dfe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ce5a9955049e99ede4a4a1ddba1dfe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ce5a9955049e99ede4a4a1ddba1dfe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ce5a9955049e99ede4a4a1ddba1dfe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f205dd87b3fbfcc5a970eb785ae49f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_24b451db28f09b97a7c2a616d8d31c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_539024b98d34cfdd70fca13258ccb040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_933c7a15fc64e9acfe053bcf955238bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4c02aa2e7e2b8a69c9c715d0d88af463(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c81629d9570df8fedf564b19ef9f49d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1be5ed938e550bd4524a9e26e34a6a18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7ec722b3d31e9c2eef30727defe23d81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_707c4582f1805582ae2baae8659ba187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([22, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f205dd87b3fbfcc5a970eb785ae49f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_539024b98d34cfdd70fca13258ccb040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_af3651cdcc34245f2e61940e659077fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37a3d5fbc8bfb5379fda2f24fc436db2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_539024b98d34cfdd70fca13258ccb040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_00ec1d1990deb6eafdee92cc8aac660e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00ec1d1990deb6eafdee92cc8aac660e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00ec1d1990deb6eafdee92cc8aac660e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00ec1d1990deb6eafdee92cc8aac660e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00ec1d1990deb6eafdee92cc8aac660e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00ec1d1990deb6eafdee92cc8aac660e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8de41fcd8473a9d3e73609e749120488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ab9739e18f9d139eee242916da0d510
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9261, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83aa8a7b45397a8db1803ba8d9d46cf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([171, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6292c60609c2c391d61219baffefc6d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6292c60609c2c391d61219baffefc6d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6292c60609c2c391d61219baffefc6d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6292c60609c2c391d61219baffefc6d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6292c60609c2c391d61219baffefc6d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6292c60609c2c391d61219baffefc6d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4e241cf876166f8be2e6079db3ad98f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ab9739e18f9d139eee242916da0d510
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 7581, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a830d89e5293b17b09dcdea93370fb56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1248], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_35ba411914695f9db2eff8349ba6e42b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1ae25568b2a11705430e09659326f860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b3cd063abfccf02a1d14c41a641973a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b3cd063abfccf02a1d14c41a641973a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b3cd063abfccf02a1d14c41a641973a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b3cd063abfccf02a1d14c41a641973a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b3cd063abfccf02a1d14c41a641973a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b3cd063abfccf02a1d14c41a641973a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fed0d16f6174f6945c1018652b7ae100(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a095fa29a69d0fc6888033ff997af5e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([145, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_539024b98d34cfdd70fca13258ccb040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fd49e629873c99d2841a697bf9a38a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e42b0def7ef6c19d7a460d0da1815912(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af28f23ef0737dad32380798a6d02c78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e42b0def7ef6c19d7a460d0da1815912
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_539024b98d34cfdd70fca13258ccb040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_24b451db28f09b97a7c2a616d8d31c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f205dd87b3fbfcc5a970eb785ae49f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_933c7a15fc64e9acfe053bcf955238bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fd49e629873c99d2841a697bf9a38a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f205dd87b3fbfcc5a970eb785ae49f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fc420555eb62fef51c17c1d1adfc99be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc420555eb62fef51c17c1d1adfc99be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc420555eb62fef51c17c1d1adfc99be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc420555eb62fef51c17c1d1adfc99be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc420555eb62fef51c17c1d1adfc99be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc420555eb62fef51c17c1d1adfc99be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_42bcc75dd0cb7ae733bba72262a3280a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d7dbc942533a350e71f544d7683d7da6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e0d651bd5d6f7b5a7d2fdded522a299e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_df28fdc0be1707bf4303ab604d96c7f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_37fa0a8929f5f8af842cc9080a03a0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_da854d1aa403749e0a6fcf76bfae50f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_85895d42e3d27f718278d29fcfad88e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bd27ef872a0f60d8722fde12abe610ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_af08d0d2c07225ae8596c30ed034501c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_95aedbdc1a1638247b9dd6c49a16ad9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61994f8677d8419ba2a134b3ac3347d
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]




if __name__ == '__main__':
    unittest.main()