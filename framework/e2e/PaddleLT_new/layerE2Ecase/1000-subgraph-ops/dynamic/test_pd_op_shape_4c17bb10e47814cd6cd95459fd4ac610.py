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



class PrimitiveOp_94bb955fee1e441052cca15c7b418076(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ad3febe7cabd5be1715d91169ab6430(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([1524, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c9f785760b6175a44cf805d7b11df078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c9f785760b6175a44cf805d7b11df078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9b5e21e6e5578b82d1d2ba8146d57465(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([2340, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8e1146593cf415e141d369d92363554f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8e1146593cf415e141d369d92363554f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_54be9492f0763127dca1a4c6d782b47c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cbbf0527ea2f860dc64fd10a4d1e45dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_875c931bfc33e9afc7fdbed679ba8642(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_812d320a82aae2edba8760c3d4fb3bab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_812d320a82aae2edba8760c3d4fb3bab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ece28a280a9f6e30f954d3368359fd71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_42530b6e8e0fa6476eef4fbe948cc3b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([2047, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc2dca7cd20614be5e51a3b4a6667dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc2dca7cd20614be5e51a3b4a6667dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_20dfecc801530a939357db440ed6b37a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e5ff0814a78828c9495d0fb4745d2f77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b51c11261b6152825eea69b2dc711a0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([1813, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3e93c3dec9d81595259d8471b2eb2638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3e93c3dec9d81595259d8471b2eb2638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_f50576d1a02dc907f560c8a820082a7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eea06d007e5ffd76a9251e78fc212b52
    def get_inputs(self):
        return [
            paddle.uniform([4875], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_289bbeb1e19d17e6dd51aae4f137bc6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([3061, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5a240a2ceea76496f474698dbb9d9268(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5a240a2ceea76496f474698dbb9d9268(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e83d7ed69805cac2314c23bfa8abac1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dee12f2b40a17b57a82b1c6532a34d73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([2062, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bbc47787d6f6f34722fb52e5f154a4b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bbc47787d6f6f34722fb52e5f154a4b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_56c98285f8110b494c7b14efd4325665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cbbf0527ea2f860dc64fd10a4d1e45dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cbbf0527ea2f860dc64fd10a4d1e45dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cbbf0527ea2f860dc64fd10a4d1e45dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ad270252b802aa2c0753431cbdfd22dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_56c98285f8110b494c7b14efd4325665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_56c98285f8110b494c7b14efd4325665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_56c98285f8110b494c7b14efd4325665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_363e942474c19aff687862fd9d8076ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ccc1643dab4c712512e7d37139c53118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([5526, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f6bf71879b65f6ecef7ced0005c38de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f6bf71879b65f6ecef7ced0005c38de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f31510cb167a7b7858b27a7529ec908(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([1071, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1bafc5319ee0ea5eae65fa82dccf2993(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1bafc5319ee0ea5eae65fa82dccf2993(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e83d7ed69805cac2314c23bfa8abac1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
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


class TestPrimitiveOp_7e3255d7561bb413500a4110e6da7a54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cf6eaa4c3318947eb9fdb8c9d94654a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([1760, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_79f11774d5d7890b86e6b1be761eb754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_79f11774d5d7890b86e6b1be761eb754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b67eaad0be681c739764a81064fddb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eea06d007e5ffd76a9251e78fc212b52
    def get_inputs(self):
        return [
            paddle.uniform([17571], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_e9e2d4307ed5a1a279070472b5f4216f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([4204, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d466908d586fbc8a6ff0d9932fc92dec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d466908d586fbc8a6ff0d9932fc92dec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2019dd7f620d99f85686fee3878ba637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
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


class TestPrimitiveOp_a087d6472332f685d97f8063fe9149b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([4680, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_53a9c31aeb113604f7f3cb8ea75f14a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_53a9c31aeb113604f7f3cb8ea75f14a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_842e7027a76b2a14b3516f376d788d9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([3778, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_654ce16003141025728ea37b8cce0433(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_654ce16003141025728ea37b8cce0433(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94bb955fee1e441052cca15c7b418076
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()