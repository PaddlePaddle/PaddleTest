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



class PrimitiveOp_25af0ca8b1099f1a176e9d4b52013b77(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 576, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a0e72f83f9a7fae8cffbe2dc5f373e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25af0ca8b1099f1a176e9d4b52013b77
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_555260b46f5d4a6a72a025806b8cdef9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e7b11a5036204283a2c5d8f1bb141f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_555260b46f5d4a6a72a025806b8cdef9
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(18496.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_1c7a4be3a8fbfed1cba7577260102b48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_892e1ccd7414c6585e5ce259ae1abb49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c7a4be3a8fbfed1cba7577260102b48
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_61a805987e1e8cb6a172636260758ae4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 21824], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_891c886972014cea2c563de2a69d0798(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61a805987e1e8cb6a172636260758ae4
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[-0.09320173412561417], [0.01596316322684288], [-0.012491357512772083], [0.005230063572525978], [-0.4879015386104584], [7.425199873978272e-05]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_60db5e4cce9fcd0b91fdb729c3a84e1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61a805987e1e8cb6a172636260758ae4
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[-0.1583651900291443], [0.06978408247232437], [-0.03330419212579727], [0.008514243178069592], [-0.12984472513198853], [0.016660870984196663]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_a4f3fbf22ef1c2a2289a76869bb7758f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61a805987e1e8cb6a172636260758ae4
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[-0.21693578362464905], [0.09848454594612122], [-0.03617676720023155], [0.03897161781787872], [-0.24818232655525208], [0.023791441693902016]]], dtype='float32').reshape([1, 6, 1]),
        ]


class PrimitiveOp_2248fee1ba62d0098c3f4d5b87e40043(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 100, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c091cbba25152ff2749cf59bff8fda6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2248fee1ba62d0098c3f4d5b87e40043
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9c091cbba25152ff2749cf59bff8fda6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2248fee1ba62d0098c3f4d5b87e40043
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_5f05a1746a091fe779f485a29168fe33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 100, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 4, 100, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_799fd4483319a2fa7e336cbaf1de6ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f05a1746a091fe779f485a29168fe33
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 100, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e7a7e07c98df6632d59175511f8874a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[100, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d84e1f1ec383e37c9619b37de1cd676e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7a7e07c98df6632d59175511f8874a8
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([100, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_148d6ac01daccbf3ef5985d4a721ccdc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e93c1897abf99f461dbf4ac740fd071(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_148d6ac01daccbf3ef5985d4a721ccdc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.49661511182785034, 0.4518659710884094, 0.1813555359840393, 0.06809335947036743], [-0.12190264463424683, 0.21905720233917236, -0.35199564695358276, 0.14050370454788208]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.37349313497543335, 0.04160088300704956, -0.3319505751132965, -0.08136159181594849], [0.08238852024078369, -0.3146577477455139, -0.4704170227050781, -0.007588326930999756]], dtype='float32').reshape([2, 4]),
        ]


class PrimitiveOp_6381ac9adbaa89dec0c7594136501af0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_13efa6dba8e367cebda8c84b7eba7347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6381ac9adbaa89dec0c7594136501af0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_13efa6dba8e367cebda8c84b7eba7347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6381ac9adbaa89dec0c7594136501af0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_41724c6d1dc408f2518b7c352b7d5d2f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1025, 1025], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 12, 1025, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b464121430e7ec9a271cb57e8f464984(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41724c6d1dc408f2518b7c352b7d5d2f
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 1025, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f89b125e636768af9b4493b2790d8594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(2005.3819580078125, dtype='float32').reshape([]),
            paddle.to_tensor(6032.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b8b9659808b4851091e60845bd094bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b8b9659808b4851091e60845bd094bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d9b6439532c9ba79072152eb89e60f87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fbbe5145672102685e72dcb696549043(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(3248.9658203125, dtype='float32').reshape([]),
            paddle.to_tensor([0.14027440547943115], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c971ce20dee48bdfdae07a1602d817fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3c9504aa787afe9bd69edebf057ee230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c971ce20dee48bdfdae07a1602d817fa
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1a1edc2185ec496e85391ec6c5b27f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(345.3778076171875, dtype='float32').reshape([]),
            paddle.to_tensor([0.14027440547943115], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d2ea16728975e761177ba7cc35477b90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(3202.2060546875, dtype='float32').reshape([]),
            paddle.to_tensor(9508.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_26efa0d1c80d669b884177b60994b803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_26efa0d1c80d669b884177b60994b803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_627f04f095163a4fe7324edf516406e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(98601.8671875, dtype='float32').reshape([]),
            paddle.to_tensor([0.23613697290420532], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5a4a46461b5b6587fd0a255bc988231b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c971ce20dee48bdfdae07a1602d817fa
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_063b631a09338cae07d62d0d4dc728fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-31.566131591796875, dtype='float32').reshape([]),
            paddle.to_tensor([0.23613697290420532], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_eceeae894259106eef06544586420874(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d669c0581e5c6cc305dbad3faf62b33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eceeae894259106eef06544586420874
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(144.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_241011960a8cb233f88498b30179eeff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f5c8c1a6d1b86e4e3b031b5618908c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_241011960a8cb233f88498b30179eeff
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8f5c8c1a6d1b86e4e3b031b5618908c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_241011960a8cb233f88498b30179eeff
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_61592fc5fc967acf1cfbd8b026ad8840(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8, 1024, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c9b57a86948d003bf58bde8b53b06f0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61592fc5fc967acf1cfbd8b026ad8840
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 1024, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_91f56cb14bb241275d35705ca5453495(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 2304, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1db042ac0604c07ad0c3217c4909edea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91f56cb14bb241275d35705ca5453495
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1db042ac0604c07ad0c3217c4909edea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91f56cb14bb241275d35705ca5453495
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_bcff2a22176319290947c71a4a6ef2fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 3136, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_371f29e24c5bcdee7bf4256a38c49f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcff2a22176319290947c71a4a6ef2fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_371f29e24c5bcdee7bf4256a38c49f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcff2a22176319290947c71a4a6ef2fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_98d9d8acbdbbc609aaec5592219aeced(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1096347d31a41a9338de90fe4a716aea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98d9d8acbdbbc609aaec5592219aeced
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(14400.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_a47274817015a7349f92259db4706444(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 3136, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_029cdfebc5a2e02580b06d90a235676f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a47274817015a7349f92259db4706444
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_029cdfebc5a2e02580b06d90a235676f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a47274817015a7349f92259db4706444
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_123458163592094d66722fa8aadcafd9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_25f47c97a4036b866c4becaa9afd2981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123458163592094d66722fa8aadcafd9
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_25f47c97a4036b866c4becaa9afd2981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123458163592094d66722fa8aadcafd9
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_78b9fbfda75fc362683e18e110f8a74c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a69e06d82f440dfa469849299ae89096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78b9fbfda75fc362683e18e110f8a74c
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a69e06d82f440dfa469849299ae89096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78b9fbfda75fc362683e18e110f8a74c
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_9bed0a8c1232d259af5d750ac1058037(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dd47e633ed68fe47dc56876f4358d2db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bed0a8c1232d259af5d750ac1058037
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_dd47e633ed68fe47dc56876f4358d2db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bed0a8c1232d259af5d750ac1058037
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_200e6629d14ef72ab61bc26745d6b3f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a8617ddb79b89aa74ff1dae2373aedbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_200e6629d14ef72ab61bc26745d6b3f2
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a8617ddb79b89aa74ff1dae2373aedbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_200e6629d14ef72ab61bc26745d6b3f2
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_65a94ce4fcadd171d195584afa2bfe7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 196, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53f4bf1462db00f34934a82df59137a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a94ce4fcadd171d195584afa2bfe7d
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_53f4bf1462db00f34934a82df59137a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a94ce4fcadd171d195584afa2bfe7d
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_e13cdb09e433b59dd353aef918254000(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ba0606e889c830d669000f7179fa084(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13cdb09e433b59dd353aef918254000
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1600.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_b1f39f603d43ab4921d299a6df37b647(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afae707c7993638537d26131333dde98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1f39f603d43ab4921d299a6df37b647
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_25faaed29840855748cd16df4fb2bdab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 197, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e1c0c171895aededff7cb4441b55797(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25faaed29840855748cd16df4fb2bdab
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8e1c0c171895aededff7cb4441b55797(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25faaed29840855748cd16df4fb2bdab
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_a68da5c2e8d7ad3c6c4635e4fcc10bc1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 197, 197], dtype='float32'),
            paddle.static.InputSpec(shape=[54, 3, 197, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e98db9450aeafb2834e115abefe6cdef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a68da5c2e8d7ad3c6c4635e4fcc10bc1
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 197], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 3, 197, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d9aa7ecb13b166316c2c5c04c93db62b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 784, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2468c61eef5edcc5d7aedb9ef8aeab54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9aa7ecb13b166316c2c5c04c93db62b
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2468c61eef5edcc5d7aedb9ef8aeab54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9aa7ecb13b166316c2c5c04c93db62b
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_64508d0b7fb630d6b59223975ab7586e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5551cf736cf2b817769ab6d7ac7884a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64508d0b7fb630d6b59223975ab7586e
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05501023307442665], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.30460914969444275], [-0.22673238813877106], [0.33773860335350037], [-0.5673317313194275]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_0f7b90be21dd940b48aece665a7c5cd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64508d0b7fb630d6b59223975ab7586e
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.26609811186790466], [0.34334731101989746], [0.3706643283367157], [0.3525908291339874]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.5707072615623474], [0.1166149377822876], [0.7084029316902161], [-0.21474090218544006]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_65d1909f343aa665e5af066ee42903a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-1.6601017713546753, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5c4dac6ec2f7998539fe40798a95b24f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 7581, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f3074acb96bd631907752b429a6af81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c4dac6ec2f7998539fe40798a95b24f
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 7581, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            paddle.static.InputSpec(shape=[6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_575ab452613b3a10df55f55944c5e9e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, -0.0, -0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([0.038921747356653214, -0.32281866669654846, -0.07439248263835907, -0.04894116520881653, 0.0072540417313575745, -0.01245079655200243], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_60bf915aea8ebe9083b93e2598435f3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20327433943748474, 0.04794470593333244, 0.01825186423957348, 0.08685724437236786, 0.38691791892051697, 0.033189114183187485], dtype='float32').reshape([6]),
            paddle.to_tensor([0.21331369876861572, 0.5722377300262451, 0.479294091463089, 0.4556185305118561, 0.4904760718345642, 0.09687614440917969], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_e5e9a7acba523caf7b50073d5ec1e6d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.23735523223876953, -0.57695072889328, -0.5512745380401611, -0.25409436225891113, -0.22113677859306335, 0.03664344549179077], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.16398099064826965, 0.5595255494117737, 0.13494634628295898, 0.19261020421981812, -0.03280341625213623, -0.33978235721588135], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_e8dab0b2f800c8c95e616fef7671f486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.18678200244903564, 0.5090863704681396, -0.39860811829566956, -0.3919111490249634, -0.19880616664886475, 0.11437070369720459], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2738617956638336, -0.008014678955078125, 0.6923106908798218, -0.5605618357658386, -0.32535621523857117, -0.5500919222831726], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_12f685853c00dd461b250ba7579875e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([0.9923672080039978, 0.23060952126979828, 0.2648046910762787, 0.9516193270683289, 0.3103187382221222, 0.0038577092345803976], dtype='float32').reshape([6]),
            paddle.to_tensor([1.9923672676086426, 1.2306095361709595, 1.264804720878601, 1.9516193866729736, 1.3103187084197998, 1.0038577318191528], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_27378b305c6918605cb8bc544e92e1e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(9.527351379394531, dtype='float32').reshape([]),
            paddle.to_tensor(6.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_62a255c61caad2c4531435061d42647e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2100, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8dcf9a8b9ea398deb796b2b9623ee020(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62a255c61caad2c4531435061d42647e
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2100, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ad9a38608293841c81cecfc6630e08ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(7.647005558013916, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9f209460593495992a15bc5caa65ec1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_87ffc4335e9ea34beef51f6851e16fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f209460593495992a15bc5caa65ec1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(28224.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b77adfe5025993ea63c577e050b8e2bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b77adfe5025993ea63c577e050b8e2bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ac5c549bcee32d1841a691ecc996542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2ac5c549bcee32d1841a691ecc996542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_d950b5f47cc8adf9a94f191208296729(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4, 1, 50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1db07e8367d70f64323e92a12f2340d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d950b5f47cc8adf9a94f191208296729
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4, 1, 50, 76], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e74f60fee84f5eb01971fec659900062(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 1, 50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4fb059da76e2d04c6a77999c45ae78ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e74f60fee84f5eb01971fec659900062
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_6a984ad88923d9ce9e3ceb8bc666c394(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 640, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3536cfa125c1fc643ad6a70bda86c5cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a984ad88923d9ce9e3ceb8bc666c394
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3536cfa125c1fc643ad6a70bda86c5cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a984ad88923d9ce9e3ceb8bc666c394
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_f56409e3719d737e0ff60edf2543900d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 640, 640], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 2, 640, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c16820b7f19fe75ffad6b58fd5ab8029(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f56409e3719d737e0ff60edf2543900d
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 640], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 2, 640, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_664ee626c9a7bc570e987b6e1ded6df1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98d9d8acbdbbc609aaec5592219aeced
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(10816.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b77adfe5025993ea63c577e050b8e2bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b77adfe5025993ea63c577e050b8e2bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_1aaaa64cbdaef9d69fa69ac5497e7945(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ebc7db99b1d61dec15d7e951f04183f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1aaaa64cbdaef9d69fa69ac5497e7945
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ad6ecad809d350691b47c24d8a26d762(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df8207840e16cfbb4239d49254255314(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad6ecad809d350691b47c24d8a26d762
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ae95baf46c66a2712f8e813f17f7f0fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25af0ca8b1099f1a176e9d4b52013b77
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(169.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_2ca4e5ab2c2ee85c9b5fa42e6272cb60(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[300, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9fbdad52a5a49066b17727e500bda521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ca4e5ab2c2ee85c9b5fa42e6272cb60
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([300, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_242fb8e56952d8390f9ef421bd3381e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_148d6ac01daccbf3ef5985d4a721ccdc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.43846940994262695, -0.03113764524459839, -0.0333956778049469, -0.23674654960632324], [-0.4677824079990387, 0.26806706190109253, -0.09477770328521729, 0.15495896339416504]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[-0.007927536964416504, -0.16195064783096313, -0.28389254212379456, 0.16157162189483643], [-0.13311100006103516, 0.06510329246520996, 0.16819244623184204, -0.4208568036556244]], dtype='float32').reshape([2, 4]),
        ]


class PrimitiveOp_9765427904702b3cec29af184420b13e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c5c68226d73ea19ad96745c18a7052b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9765427904702b3cec29af184420b13e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2704.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_b52152804f4242ccda695798e1a24ed3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4, 1, 100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d63dfa7725a4d6ef6a3ebc6e92664e0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b52152804f4242ccda695798e1a24ed3
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4, 1, 100, 152], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_798db199564304f5cd70c6a68423a2c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 1, 100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a41c89fe1695cce953362f02cfabce12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_798db199564304f5cd70c6a68423a2c5
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_07078f5e631eff26eba52f1d8b40799b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21, 16384], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 21, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20eb74bb77a7c21c90910ec67aefb7ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07078f5e631eff26eba52f1d8b40799b
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 16384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[10319.40234375], [10370.044921875], [10332.4931640625], [10393.3369140625], [10320.5810546875], [10317.306640625], [10364.3837890625], [10340.9453125], [10314.767578125], [10371.3271484375], [10360.5693359375], [10320.3720703125], [10301.119140625], [10339.4951171875], [10327.345703125], [10360.9296875], [10350.373046875], [10360.7158203125], [10313.0107421875], [10357.9189453125], [10365.0478515625]]], dtype='float32').reshape([1, 21, 1]),
        ]


class PrimitiveOp_849ec7c645ad4a347bd586ebdbc99fcc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c9c4143af5c5597fcc1f13bbfeb25a29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_849ec7c645ad4a347bd586ebdbc99fcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_371f29e24c5bcdee7bf4256a38c49f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcff2a22176319290947c71a4a6ef2fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_371f29e24c5bcdee7bf4256a38c49f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcff2a22176319290947c71a4a6ef2fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_e991b1e1d68b589453cda60b5040c397(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 200, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afb18cb965c43c507f85f1b287bc864f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e991b1e1d68b589453cda60b5040c397
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_afb18cb965c43c507f85f1b287bc864f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e991b1e1d68b589453cda60b5040c397
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_ecd5aa0821478e2032e0886ad088d9e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 200, 200], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 2, 200, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_31d259fdbdd774ec53ef8e915bb3ca36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecd5aa0821478e2032e0886ad088d9e8
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 2, 200, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_324a042b2e220af09c5477a86aeb91d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_324a042b2e220af09c5477a86aeb91d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_71acdc1388cce56742292403d11b605d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13cdb09e433b59dd353aef918254000
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4096.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_8357bb4e351fddef99148117a3fc5989(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7f0d01fe77a2a019a25f02d11de28d64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8357bb4e351fddef99148117a3fc5989
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.03849673643708229], [0.02545531839132309], [0.47374671697616577], [0.0524928979575634], [-0.030130885541439056], [-0.03007093444466591]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_13296870fc8d9c7fe6e96893493f07c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8357bb4e351fddef99148117a3fc5989
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.6128764748573303], [-0.12078536301851273], [-0.1835383176803589], [-0.057387858629226685], [0.16328364610671997], [0.040456049144268036]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.5743797421455383], [-0.09533004462718964], [0.2902083992958069], [-0.004894960671663284], [0.1331527680158615], [0.010385116562247276]], dtype='float32').reshape([6, 1]),
        ]


class PrimitiveOp_89faa8f9cfceb6a68c0b564dd96973be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9261, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc45fd7f184ed5b47a89609c5e0ad127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89faa8f9cfceb6a68c0b564dd96973be
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9261, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_dffe5ee239f91530cab66dd52f24ef43(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 196, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5866e07ffc06e786e8cc63a7a8559fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dffe5ee239f91530cab66dd52f24ef43
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5866e07ffc06e786e8cc63a7a8559fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dffe5ee239f91530cab66dd52f24ef43
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_48b9aa3cd446aa0bbd2be50fb70662d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4116, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_025a0af5aa7f376da763cc6ff7c000fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48b9aa3cd446aa0bbd2be50fb70662d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4116, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c86e7ae0c322e377577b1a83148500f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9765427904702b3cec29af184420b13e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1600.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2794f233f52bd7c24e3e81875da463e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2794f233f52bd7c24e3e81875da463e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_d218101ae6cc2c945447d4e5f3be809b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4, 1, 7, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b563b0359625a6bd32da4f58fcefda06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d218101ae6cc2c945447d4e5f3be809b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4, 1, 7, 10], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_608122f3fc89908f538ec0234f0b8d78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 1, 7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e927a4ddbcfcaf0aee587ea9125bc47d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_608122f3fc89908f538ec0234f0b8d78
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6ac05ad983416ae2ba1bf1d5a370574f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(400.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_04d88777bb5cb50fb086f6eb864923e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13cdb09e433b59dd353aef918254000
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(676.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_d20301e725e12d2a28055c2e7c0e5243(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 60800, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4cb5e0d7f3a7543cbb70c92d31ffa030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d20301e725e12d2a28055c2e7c0e5243
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4cb5e0d7f3a7543cbb70c92d31ffa030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d20301e725e12d2a28055c2e7c0e5243
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4c93e703bb68b6e2ca766cb0487d6aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4c93e703bb68b6e2ca766cb0487d6aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_029cdfebc5a2e02580b06d90a235676f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a47274817015a7349f92259db4706444
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_029cdfebc5a2e02580b06d90a235676f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a47274817015a7349f92259db4706444
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_eb1495e94fb16d27855af6b6599266b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf4c9ff9eb35790557ff2378ce0d8749(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb1495e94fb16d27855af6b6599266b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e714cc16ce2ccdf5566e9d0436e6f3de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 784, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_048890a093f8fae44ee60cdb2d2208ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e714cc16ce2ccdf5566e9d0436e6f3de
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_048890a093f8fae44ee60cdb2d2208ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e714cc16ce2ccdf5566e9d0436e6f3de
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_353035a67891e73e0688bf38c37d5a28(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21504, 1, 91], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 21504, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_061ee16774935beeafa8fccdd80972a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_353035a67891e73e0688bf38c37d5a28
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 1, 91], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 21504, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c536d82721744cc0af3ecbee4d7235a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c536d82721744cc0af3ecbee4d7235a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_471753e5a1f43b1877e5fa9a2038ad6e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 9216, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bfd3598d0cfa0438d6afb30cd7f346e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471753e5a1f43b1877e5fa9a2038ad6e
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bfd3598d0cfa0438d6afb30cd7f346e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471753e5a1f43b1877e5fa9a2038ad6e
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_f5519ba142a258f164da420bf7b4c1d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e3b3875a7f8c285245f5a8392b6a968(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5519ba142a258f164da420bf7b4c1d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(441.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_210b6b79c731e4aba93f558e9047993b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20267, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[20267, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f63516d98502bf921317be94db2215c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_210b6b79c731e4aba93f558e9047993b
    def get_inputs(self):
        return [
            paddle.uniform([20267, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([20267, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_76e2bd3d4a1e4c8208b41a22bd34f984(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2116.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_14b93159e17197c32324b6eb0782cf5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(0.6553301811218262, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5866e07ffc06e786e8cc63a7a8559fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dffe5ee239f91530cab66dd52f24ef43
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5866e07ffc06e786e8cc63a7a8559fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dffe5ee239f91530cab66dd52f24ef43
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2ac5c549bcee32d1841a691ecc996542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2ac5c549bcee32d1841a691ecc996542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6db9cbb824dce374882339b92a19e25a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98d9d8acbdbbc609aaec5592219aeced
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(65536.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_029cdfebc5a2e02580b06d90a235676f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a47274817015a7349f92259db4706444
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_029cdfebc5a2e02580b06d90a235676f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a47274817015a7349f92259db4706444
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_39d9ef8c2243d1b1e1d36f72628e322b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 128, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4212022b14cf8413d76b3679631d6d22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39d9ef8c2243d1b1e1d36f72628e322b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.4925087094306946], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_900e53729608569f217c410830f2681b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9765427904702b3cec29af184420b13e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(6400.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_9288920304028b1ae986841ae437f48b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cdfff13bd38db149b8bec0edba751f69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9288920304028b1ae986841ae437f48b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2ac5c549bcee32d1841a691ecc996542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2ac5c549bcee32d1841a691ecc996542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ce8e564d8db99a9774470bf0ff128f93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(169.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_6788cd52d4620f93775337d648c1c774(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 160, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9cccd621cc3421138f9edabf34494595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6788cd52d4620f93775337d648c1c774
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9cccd621cc3421138f9edabf34494595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6788cd52d4620f93775337d648c1c774
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_1c67356afeef3c6662fa69cb09d4b9e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 160, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 8, 160, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9412f50e90b6e3817b4d776a383f4d7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c67356afeef3c6662fa69cb09d4b9e1
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 8, 160, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_293e05703d68ce7a3d9e5544460dfa48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eceeae894259106eef06544586420874
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_d76b5ec1e1c4784a6ec33688d260f4c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a335037e6b6da03dd192d43505998719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76b5ec1e1c4784a6ec33688d260f4c9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_259a7c6c025cc531f64f7e4687c7ab22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 169, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_85b1700baa4d50eb732de4f8ba1ffd35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_259a7c6c025cc531f64f7e4687c7ab22
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_85b1700baa4d50eb732de4f8ba1ffd35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_259a7c6c025cc531f64f7e4687c7ab22
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_85b1700baa4d50eb732de4f8ba1ffd35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_259a7c6c025cc531f64f7e4687c7ab22
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_85b1700baa4d50eb732de4f8ba1ffd35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_259a7c6c025cc531f64f7e4687c7ab22
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9466a158408d010e6204432df8a45701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(2703.76708984375, dtype='float32').reshape([]),
            paddle.to_tensor(8060.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_56ae1376f476c2a5a62af1d301361060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_56ae1376f476c2a5a62af1d301361060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_edf301ca82655fdaa737928225e55877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(2108.522216796875, dtype='float32').reshape([]),
            paddle.to_tensor([0.09485089778900146], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_45385e7412d10946b73c8ab85a27e362(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c971ce20dee48bdfdae07a1602d817fa
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_17ebf5198a0fb25d8e3a4b60b48aa914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-86.49337768554688, dtype='float32').reshape([]),
            paddle.to_tensor([0.09485089778900146], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_13ef74aa9f50b42fe3d36a1229797140(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cff609ee882a2a654110d543d9fd4b5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ef74aa9f50b42fe3d36a1229797140
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7ca25a97a8819c2aac095142f2b4ff7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eceeae894259106eef06544586420874
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(529.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_92701321a6b7e6725c5d1a71eb0efc27(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32768, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de86c7793f74724dd11127e1f11b852a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92701321a6b7e6725c5d1a71eb0efc27
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(32.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_de86c7793f74724dd11127e1f11b852a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92701321a6b7e6725c5d1a71eb0efc27
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(32.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_c7ac828c055996baae99cc2e1e97ae67(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 320, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8032a7476c099fb04877e5b6bce55d1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ac828c055996baae99cc2e1e97ae67
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8032a7476c099fb04877e5b6bce55d1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ac828c055996baae99cc2e1e97ae67
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_e8a91bff8f42093c941ca58c9512caf2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 320, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 4, 320, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1663b2787ef0db1dde2e407c28723a82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8a91bff8f42093c941ca58c9512caf2
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ef3619de02b323bad3de8a88a2380e0d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d10374948246dbdc8c79d52bcb42ba18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef3619de02b323bad3de8a88a2380e0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fa4e7eecebba71d0c04eafd01c953090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9765427904702b3cec29af184420b13e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(9216.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_42844b690326291e4cdd8ea95897c54d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_369a28e5bd1118559452fb793bdf0f8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42844b690326291e4cdd8ea95897c54d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b9f61dd232927c655bbcbb098b56d50c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[2, 1, 960, 960], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d5bfb975d2627f2446fea01ed0b5077a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9f61dd232927c655bbcbb098b56d50c
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4943d5957299860748a46c4b99879007(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8400, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[8400, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f71a7634769cdc132273c797da6838d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4943d5957299860748a46c4b99879007
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([8400, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f2c1de82a6a9883ab9bbd8cb93d93854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9765427904702b3cec29af184420b13e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(12544.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_7c84f2d13b6bd2bd2f44408312c90b78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b20c5e402af342282b6d1eb2d37e15f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c84f2d13b6bd2bd2f44408312c90b78
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.4610447883605957], [0.07641894370317459], [-0.04706235229969025], [-0.006878872402012348], [-0.33857452869415283]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_fab76f93b4aa5481aa52ac26e865a6b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c84f2d13b6bd2bd2f44408312c90b78
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08270445466041565], [0.10558044165372849], [0.0877026915550232], [0.015834761783480644], [0.3873971700668335]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.37834033370018005], [0.18199938535690308], [0.04064033553004265], [0.008955889381468296], [0.048822641372680664]], dtype='float32').reshape([5, 1]),
        ]


class PrimitiveOp_e7dfc93ce6288b0a6324e22fcb56a08e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 65536, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f8dc0f39877fdb553b9ec07c4e12c53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7dfc93ce6288b0a6324e22fcb56a08e
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(32.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9f8dc0f39877fdb553b9ec07c4e12c53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7dfc93ce6288b0a6324e22fcb56a08e
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(32.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a0b0e0970b1fdefc53123adeff22f30d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(2447.673828125, dtype='float32').reshape([]),
            paddle.to_tensor(7320.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_92c0030c558d2428ce2489a4e6ad0513(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_92c0030c558d2428ce2489a4e6ad0513(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3bd15dda845fcc6eb8183b76376926b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(39225.9609375, dtype='float32').reshape([]),
            paddle.to_tensor([0.22811788320541382], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_40c7b05f67c02da6e4eed76f77c31d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c971ce20dee48bdfdae07a1602d817fa
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3399afae035c8b5c751a1c238b33231d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(379.978271484375, dtype='float32').reshape([]),
            paddle.to_tensor([0.22811788320541382], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fee08ae21eeb4469894dbe88db6ed719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13cdb09e433b59dd353aef918254000
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1156.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_3546e8dc8a6842daa2ebd91dc0d08e39(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_910e7e08fb820db8de0d850e1c479db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3546e8dc8a6842daa2ebd91dc0d08e39
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_910e7e08fb820db8de0d850e1c479db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3546e8dc8a6842daa2ebd91dc0d08e39
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_99603eb965a06cc637c6f16eebe6c271(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8, 512, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d72ac10bd2f917af9d83893dec134653(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99603eb965a06cc637c6f16eebe6c271
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_09f9c9507da7244963925ab3fcd8d64f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 19, 34], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9cdb6da65f9c6fe3afd49c4109ab1737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09f9c9507da7244963925ab3fcd8d64f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ec56cb2a6f06f470ec2eba064972bba2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 2304, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c3f56c70f0fcfaec601e67975bf2049(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec56cb2a6f06f470ec2eba064972bba2
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8c3f56c70f0fcfaec601e67975bf2049(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec56cb2a6f06f470ec2eba064972bba2
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_39277b25967a2db522cbf9baee8c7c84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21760, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f29219c4942441f0563d878b0aa49c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39277b25967a2db522cbf9baee8c7c84
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f29219c4942441f0563d878b0aa49c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39277b25967a2db522cbf9baee8c7c84
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_dae5a2d74f6dfdf3c08830dab354a8c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9765427904702b3cec29af184420b13e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(65536.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_6045089f2062055bbc32b22cffcac536(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 198, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00bae2f51fd1dcf8acd9da63cc47488f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6045089f2062055bbc32b22cffcac536
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_00bae2f51fd1dcf8acd9da63cc47488f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6045089f2062055bbc32b22cffcac536
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_929a59320a8ce308a60b65bc64b15fd3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 198, 198], dtype='float32'),
            paddle.static.InputSpec(shape=[54, 3, 198, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_87b7371ede55879b65ee925c56ccf056(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_929a59320a8ce308a60b65bc64b15fd3
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 198], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 3, 198, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f3d856a6003b5cf89f06eeb0f0e321b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(361.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_574f977a1d9623aef2b1dc391937dcc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(4088.576171875, dtype='float32').reshape([]),
            paddle.to_tensor(12156.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_489832da5d38e98e2902cc85a547bec8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_489832da5d38e98e2902cc85a547bec8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8bd1a4fee6cb3f25e36943074bd8d95d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(7003.640625, dtype='float32').reshape([]),
            paddle.to_tensor([0.40433353185653687], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_606bcf59fb87bcc2a13a050ea816c26f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c971ce20dee48bdfdae07a1602d817fa
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_32ac1860a1488ef2848a500b0613da2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-396.4815673828125, dtype='float32').reshape([]),
            paddle.to_tensor([0.40433353185653687], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_65534a5001d55ed4b298546ec18dff16(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b736e7b4b0a8658962e75658a289a63c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65534a5001d55ed4b298546ec18dff16
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(16384.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_c0ae64e72e5aa4e75259ac44863cd3ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_61e7931d86ec467c5b0e4630a2fa5050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0ae64e72e5aa4e75259ac44863cd3ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_81bf2f1cee6052f714b65115c017329d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3549, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_feaa8016dd66d455fa3594dd3e3fc1fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81bf2f1cee6052f714b65115c017329d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3549, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bcc50d1910fe1297f2cb93d3e8f34ba0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_241011960a8cb233f88498b30179eeff
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bcc50d1910fe1297f2cb93d3e8f34ba0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_241011960a8cb233f88498b30179eeff
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c9b57a86948d003bf58bde8b53b06f0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61592fc5fc967acf1cfbd8b026ad8840
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 1024, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_048890a093f8fae44ee60cdb2d2208ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e714cc16ce2ccdf5566e9d0436e6f3de
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_048890a093f8fae44ee60cdb2d2208ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e714cc16ce2ccdf5566e9d0436e6f3de
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4a10e8b2dafe4aa606884349437207ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13cdb09e433b59dd353aef918254000
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(16384.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_b264268af78d24055b659edf41a403e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a4f9b02baaf4a3469969ce5516c39ee3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b264268af78d24055b659edf41a403e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.3002668619155884], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c8daed9b9727979c396ac071efc66db1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_18f0916e02c2e9c0a93228706b5e3167(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8daed9b9727979c396ac071efc66db1
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2704.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_324a042b2e220af09c5477a86aeb91d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_324a042b2e220af09c5477a86aeb91d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f048a6f81b6a92fc4feab114cdb775b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9765427904702b3cec29af184420b13e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4624.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9c091cbba25152ff2749cf59bff8fda6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2248fee1ba62d0098c3f4d5b87e40043
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9c091cbba25152ff2749cf59bff8fda6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2248fee1ba62d0098c3f4d5b87e40043
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_477836b2efc66ca1e0c3afd9e02f7ad3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59a84f89e49c56027f941f3c08b0bc08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_477836b2efc66ca1e0c3afd9e02f7ad3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.4338390827178955], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d6d673aa53487facfa88844d66a76af7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20465cdd44074d4ea93633b2d8923c81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6d673aa53487facfa88844d66a76af7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.07617727667093277], [-0.028213033452630043], [0.13593381643295288], [0.1629621386528015], [-0.42407214641571045], [0.11530013382434845], [-0.39416757225990295], [-0.016913022845983505], [0.24559549987316132]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_8d66227afeaee94bd372c436e579ab53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6d673aa53487facfa88844d66a76af7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0800543949007988], [0.2637481689453125], [-0.1386985182762146], [-0.12993383407592773], [0.409182608127594], [0.012348398566246033], [0.3850596249103546], [0.049205586314201355], [-0.2325476109981537]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0038771205581724644], [0.235535129904747], [-0.002764695091173053], [0.033028312027454376], [-0.01488953921943903], [0.12764853239059448], [-0.009107938036322594], [0.03229256346821785], [0.013047887943685055]], dtype='float32').reshape([9, 1]),
        ]


class PrimitiveOp_dcbaccedab1fbd8c70d4a7734e4d031f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 19, 32768], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 19, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_22eb0d4a21cb5654061031444c4d4b7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcbaccedab1fbd8c70d4a7734e4d031f
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 32768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[20670.05078125], [20725.490234375], [20732.693359375], [20756.80078125], [20722.5390625], [20699.37109375], [20725.27734375], [20691.828125], [20743.4921875], [20657.287109375], [20726.693359375], [20719.939453125], [20756.384765625], [20677.787109375], [20719.29296875], [20696.12890625], [20740.365234375], [20776.830078125], [20720.12890625]]], dtype='float32').reshape([1, 19, 1]),
        ]


class TestPrimitiveOp_bfd3598d0cfa0438d6afb30cd7f346e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471753e5a1f43b1877e5fa9a2038ad6e
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bfd3598d0cfa0438d6afb30cd7f346e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471753e5a1f43b1877e5fa9a2038ad6e
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_db88c289ab42b30cb82c68bb2cb080d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8daed9b9727979c396ac071efc66db1
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(3600.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_cb2c9b3aa1051c2cf862c2846a4f657b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74f7a73030d8f4322bcb7ede250fdab6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb2c9b3aa1051c2cf862c2846a4f657b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.33030200004577637], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e7d1c33a076036b9c76f1b2a034eff29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e7d1c33a076036b9c76f1b2a034eff29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_36859481ac5b0a1b864905ebaaee1bbe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 50, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_700586d765b3651572b907d33bc2eaee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36859481ac5b0a1b864905ebaaee1bbe
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_700586d765b3651572b907d33bc2eaee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36859481ac5b0a1b864905ebaaee1bbe
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d7259ed25c21573c6f2b68bc8aee9a6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_555260b46f5d4a6a72a025806b8cdef9
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(6400.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2468c61eef5edcc5d7aedb9ef8aeab54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9aa7ecb13b166316c2c5c04c93db62b
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2468c61eef5edcc5d7aedb9ef8aeab54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9aa7ecb13b166316c2c5c04c93db62b
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e302ee1bded171a350772ef88ae8a936(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(2725.442138671875, dtype='float32').reshape([]),
            paddle.to_tensor(8184.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_52ba020d55f97b88d5569ed795f651cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_52ba020d55f97b88d5569ed795f651cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c05bd23e7c9a76a19a5af9ed06d520e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(193.12008666992188, dtype='float32').reshape([]),
            paddle.to_tensor([-0.16134828329086304], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a5e49edbfc86cfcaf9bc5ac2ea6e571e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c971ce20dee48bdfdae07a1602d817fa
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ae4b8cc7a8aa4a941b6fffa5af3d16ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(104.61871337890625, dtype='float32').reshape([]),
            paddle.to_tensor([-0.16134828329086304], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d10374948246dbdc8c79d52bcb42ba18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef3619de02b323bad3de8a88a2380e0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c9c4143af5c5597fcc1f13bbfeb25a29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_849ec7c645ad4a347bd586ebdbc99fcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6c726bb1b70d4e89232089767898c32f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_555260b46f5d4a6a72a025806b8cdef9
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(65536.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_7045764ff1606a1c2114ef7d98ab1c2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c086e20239c0eab988be062a81709fc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7045764ff1606a1c2114ef7d98ab1c2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_18e226e95ec5d5c9402c294fc5bdc983(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13cdb09e433b59dd353aef918254000
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2704.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_53f4bf1462db00f34934a82df59137a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a94ce4fcadd171d195584afa2bfe7d
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_53f4bf1462db00f34934a82df59137a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a94ce4fcadd171d195584afa2bfe7d
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_9935249d185b359b70b3eaa5cb664cdf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_88ad5af877d7ae7ac81818d02e68eafb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9935249d185b359b70b3eaa5cb664cdf
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_88ad5af877d7ae7ac81818d02e68eafb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9935249d185b359b70b3eaa5cb664cdf
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_ec13f3817562484c7fc8cc9a39ff8bce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1174, 1174], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 12, 1174, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_446b04b1837d2c5593def4aa5ccde2a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec13f3817562484c7fc8cc9a39ff8bce
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 1174, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5520a993be93fa1c89fc6ffcb396f317(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4725, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b6aef4037934d46de62b2ce61b37a99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5520a993be93fa1c89fc6ffcb396f317
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4725, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0928d1a760bb9d25cd90186c07cd625e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_465b67050c3c534fe14d458e3bb0ce15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0928d1a760bb9d25cd90186c07cd625e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_20eb0ae327ac11fa0bc84b02e19d0b17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-2.8528876304626465, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2065899fc5ae213d249088920bfc29ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 288, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_42d163917b0f74633d2d839e442f073e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2065899fc5ae213d249088920bfc29ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(676.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_d6cc222d6b5f0acdf73a312098603926(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 11109, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c6bb989f053e65d1544847c5234da2ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6cc222d6b5f0acdf73a312098603926
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 11109, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6cd0d6b6c47c653f54b0656d8b0e756d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 198, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b24a0fa94d38bbdc9b27551a5c7b0cc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cd0d6b6c47c653f54b0656d8b0e756d
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b24a0fa94d38bbdc9b27551a5c7b0cc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cd0d6b6c47c653f54b0656d8b0e756d
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_f0d6cf9ea9c501de534a291dafa36f94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 198, 198], dtype='float32'),
            paddle.static.InputSpec(shape=[86, 3, 198, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c128a8fc567fdcd8e10bdf82f09d949(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0d6cf9ea9c501de534a291dafa36f94
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 198], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 3, 198, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_de7abdafbdfbb78ac7fbbae0f9129709(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ba01e28ec794de5339a9e52598ebd275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de7abdafbdfbb78ac7fbbae0f9129709
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4096.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a6fb96f392ea15a3c9a5434025b43e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25af0ca8b1099f1a176e9d4b52013b77
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(225.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_71bc27fdc515f7066a0529ee95b67e66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(676.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2ac5c549bcee32d1841a691ecc996542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2ac5c549bcee32d1841a691ecc996542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_743bbe2fb07fa125aa22761ab376d4ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-0.759699285030365, dtype='float32').reshape([]),
            paddle.to_tensor([3.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e01ae197afd5aa161f2954e101347f83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 9216, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4fdc5d7feb41d1d34783831e2f42cb45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e01ae197afd5aa161f2954e101347f83
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4fdc5d7feb41d1d34783831e2f42cb45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e01ae197afd5aa161f2954e101347f83
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_43a8264387599a41d2d64444ae2a10fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2065899fc5ae213d249088920bfc29ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(900.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f81a275e11e5a83f053f0f79cd857a50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(784.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_16d8bd42f7eff78b9e3230066b7a7608(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[12096, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[12096, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8294ecbef09416e8a627065175dc9159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16d8bd42f7eff78b9e3230066b7a7608
    def get_inputs(self):
        return [
            paddle.uniform([12096, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([12096, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b17c511cd99e1d7e78f1258107103a16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(576.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_94a67a6b88168f05f1fe8eb9f0eb46ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d090f9e82b68fabd1bb30894e3b535fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94a67a6b88168f05f1fe8eb9f0eb46ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6d32915e6cdc772607e041514b8f38a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_32757673a5de7770d7814034a17a11d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d32915e6cdc772607e041514b8f38a8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0671bca000ad30363b482cf27fed5a16(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 8192], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8192, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b9bb3ffd940f6c74c8bea8fe184c748(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0671bca000ad30363b482cf27fed5a16
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 8192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8192, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1b5503add412187f8e75c884e9eb3420(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a2731010d52bc6b7249ceb799e128aae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b5503add412187f8e75c884e9eb3420
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.029805947095155716]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_9baeab8f3f9884c9de597da352c18149(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b5503add412187f8e75c884e9eb3420
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2179727405309677]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.1881667971611023]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_4c93e703bb68b6e2ca766cb0487d6aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4c93e703bb68b6e2ca766cb0487d6aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c536d82721744cc0af3ecbee4d7235a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c536d82721744cc0af3ecbee4d7235a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_700586d765b3651572b907d33bc2eaee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36859481ac5b0a1b864905ebaaee1bbe
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_700586d765b3651572b907d33bc2eaee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36859481ac5b0a1b864905ebaaee1bbe
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_7019d2e489022fc2063bc30d027b50b0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 50, 50], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 8, 50, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02a501727237e9c8c22a0977711bdf88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7019d2e489022fc2063bc30d027b50b0
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 8, 50, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7a621c5d797a61a537bd3d1833d32e41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6069, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[6069, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a68c293f8857515be1b9709e6696eb7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a621c5d797a61a537bd3d1833d32e41
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6069, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_32757673a5de7770d7814034a17a11d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d32915e6cdc772607e041514b8f38a8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_53f4bf1462db00f34934a82df59137a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a94ce4fcadd171d195584afa2bfe7d
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_53f4bf1462db00f34934a82df59137a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a94ce4fcadd171d195584afa2bfe7d
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_df2d5832903d9027b874ccd5d4ce9918(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_555260b46f5d4a6a72a025806b8cdef9
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(25600.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_046d00e99ce2fe931d0733ca5796b219(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 576, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e312d8e9e4602e57d44bc4329dc3161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_046d00e99ce2fe931d0733ca5796b219
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5e312d8e9e4602e57d44bc4329dc3161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_046d00e99ce2fe931d0733ca5796b219
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4d20f3a7d7b167039795995ba02b2d0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6381ac9adbaa89dec0c7594136501af0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4d20f3a7d7b167039795995ba02b2d0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6381ac9adbaa89dec0c7594136501af0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_ee6de3d37a61c243380b5404215698a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1025, 1025], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6, 1025, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_86f49df1872087bee30d1c01098f08fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee6de3d37a61c243380b5404215698a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6, 1025, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b2c23d755f2224cc7ffc2d2609055ea3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ba9981dd68ba200d455f6a39c254ed24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c23d755f2224cc7ffc2d2609055ea3
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ba9981dd68ba200d455f6a39c254ed24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c23d755f2224cc7ffc2d2609055ea3
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_519407c9823bdee6d6b86feab6d11c70(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7bf8c384aaef90ca77b0c71455736047(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_519407c9823bdee6d6b86feab6d11c70
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7bf8c384aaef90ca77b0c71455736047(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_519407c9823bdee6d6b86feab6d11c70
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_39d555b3e5c8721d2d6273764738e2a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3db7e54fc98702a6a719d370673b455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39d555b3e5c8721d2d6273764738e2a5
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c3db7e54fc98702a6a719d370673b455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39d555b3e5c8721d2d6273764738e2a5
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_04febfe6e5f20f763ca0126890a31c04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d347c6abc84c6222fee5e77bafcb8927(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04febfe6e5f20f763ca0126890a31c04
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d347c6abc84c6222fee5e77bafcb8927(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04febfe6e5f20f763ca0126890a31c04
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cff609ee882a2a654110d543d9fd4b5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ef74aa9f50b42fe3d36a1229797140
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_98df7e833d22dd8209df20d8344410e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2065899fc5ae213d249088920bfc29ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4096.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d62f6934c7efb373edc79b7c2f78c9af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65534a5001d55ed4b298546ec18dff16
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(7056.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f270182a9c1197509346c64a0bacae73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(0.376983106136322, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_477f0876f75200ca93839878e37d3bce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_892e1ccd7414c6585e5ce259ae1abb49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c7a4be3a8fbfed1cba7577260102b48
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bb75617dbecd52f69e4af79abc406f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(7310.369140625, dtype='float32').reshape([]),
            paddle.to_tensor(21992.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_816db479581cc7008abc9c0b96ec823b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_816db479581cc7008abc9c0b96ec823b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_efa62da58cd799bf1910c1d134db1857(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-9654.203125, dtype='float32').reshape([]),
            paddle.to_tensor([0.3436046242713928], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_10d27acc9d8a69f85a2fe8fdc5746681(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c971ce20dee48bdfdae07a1602d817fa
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5b35869aced0e0aa94dd40ad9ef32f48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(36.37746810913086, dtype='float32').reshape([]),
            paddle.to_tensor([0.3436046242713928], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_418dd6ed100920388bc95f00852dcd3d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8400, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc7adaa6c09a9e8d089037733b46d970(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_418dd6ed100920388bc95f00852dcd3d
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8400, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_786f7525ab558f95884fa470895c2967(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de7abdafbdfbb78ac7fbbae0f9129709
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1764.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a335037e6b6da03dd192d43505998719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76b5ec1e1c4784a6ec33688d260f4c9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7434062c8ceb998dd2dc7cf6c572d719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5519ba142a258f164da420bf7b4c1d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d090f9e82b68fabd1bb30894e3b535fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94a67a6b88168f05f1fe8eb9f0eb46ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_559342d51f05875ada23bf1d997b9c20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(1450.2685546875, dtype='float32').reshape([]),
            paddle.to_tensor(4296.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2c75e5d767a2ddaa18e838ac92f8d549(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2c75e5d767a2ddaa18e838ac92f8d549(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d70216b89b6fb38725369ebcb5511d76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(9116.337890625, dtype='float32').reshape([]),
            paddle.to_tensor([-0.4071309566497803], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3832adfc857e828dab64ce84fec02348(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c971ce20dee48bdfdae07a1602d817fa
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ee7c97a1c66ac35b94123c554f75e10c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(72.78530883789062, dtype='float32').reshape([]),
            paddle.to_tensor([-0.4071309566497803], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e7d1c33a076036b9c76f1b2a034eff29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e7d1c33a076036b9c76f1b2a034eff29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_db51c700ab2d532ab2224306a6b610bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(2369.48291015625, dtype='float32').reshape([]),
            paddle.to_tensor(7092.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_91f0bb2186e1dc5bb061ea8123cb3a00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_91f0bb2186e1dc5bb061ea8123cb3a00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f026880ca345a7d6b0e031caa869b5d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(3520.421142578125, dtype='float32').reshape([]),
            paddle.to_tensor([0.3501899242401123], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e20925342e480efc796cfa88c48936f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c971ce20dee48bdfdae07a1602d817fa
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8895fb20ce1f17372f304bf211607c54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-463.76104736328125, dtype='float32').reshape([]),
            paddle.to_tensor([0.3501899242401123], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_231bf319a3acb2a59a477d9330779724(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 197, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf2d82e3c734412d6c7a771722a1cb49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_231bf319a3acb2a59a477d9330779724
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bf2d82e3c734412d6c7a771722a1cb49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_231bf319a3acb2a59a477d9330779724
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_f20a422e1c3323253e1ff0e283d03bc7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 197, 197], dtype='float32'),
            paddle.static.InputSpec(shape=[86, 3, 197, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_25dd2d6e9124384fb8208ef07ef10eb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f20a422e1c3323253e1ff0e283d03bc7
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 197], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 3, 197, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0f95f83b55bcb2932dc1bc1ca6ce6dbf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4, 1, 13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b5bb847fd2c4f64ed450a5170af13ba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f95f83b55bcb2932dc1bc1ca6ce6dbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4, 1, 13, 19], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_23f66bd6b8f32ce0dd944bb00acf034a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 1, 13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_903d771fb9b98c39b7ba6f54396deb92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23f66bd6b8f32ce0dd944bb00acf034a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b1e932786102ead50e7a80392afaeb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(289.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_afae707c7993638537d26131333dde98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1f39f603d43ab4921d299a6df37b647
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6771823370743869100da560aecbd0cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eceeae894259106eef06544586420874
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(169.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_3df199e5e72b5d438cad6b0fcf58582e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 4096], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4096, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9d68873c9ee567ffd8d65e8dc0b61225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3df199e5e72b5d438cad6b0fcf58582e
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 4096], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4096, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2ebc7db99b1d61dec15d7e951f04183f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1aaaa64cbdaef9d69fa69ac5497e7945
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_25f47c97a4036b866c4becaa9afd2981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123458163592094d66722fa8aadcafd9
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_25f47c97a4036b866c4becaa9afd2981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123458163592094d66722fa8aadcafd9
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a69e06d82f440dfa469849299ae89096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78b9fbfda75fc362683e18e110f8a74c
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a69e06d82f440dfa469849299ae89096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78b9fbfda75fc362683e18e110f8a74c
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_dd47e633ed68fe47dc56876f4358d2db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bed0a8c1232d259af5d750ac1058037
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_dd47e633ed68fe47dc56876f4358d2db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bed0a8c1232d259af5d750ac1058037
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a8617ddb79b89aa74ff1dae2373aedbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_200e6629d14ef72ab61bc26745d6b3f2
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a8617ddb79b89aa74ff1dae2373aedbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_200e6629d14ef72ab61bc26745d6b3f2
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_2c1f6d046045d5904bba7e9c1c196f8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 577, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fdd1357fd8c9eccfb96a79a5559d144a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c1f6d046045d5904bba7e9c1c196f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fdd1357fd8c9eccfb96a79a5559d144a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c1f6d046045d5904bba7e9c1c196f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_07fe5c32a9e045d6e17f00b555b92d04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 577, 577], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 12, 577, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_879377e6c2b88ea4f84eae3bd79ff3fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07fe5c32a9e045d6e17f00b555b92d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 577, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2936d2e8c7ea2429e57d339680e327ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-2.1168315410614014, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_53f4bf1462db00f34934a82df59137a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a94ce4fcadd171d195584afa2bfe7d
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_53f4bf1462db00f34934a82df59137a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a94ce4fcadd171d195584afa2bfe7d
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bf4c9ff9eb35790557ff2378ce0d8749(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb1495e94fb16d27855af6b6599266b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_05d21a58ad0b272031b0bd2636ffef17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13cdb09e433b59dd353aef918254000
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(3136.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_d082bdf7603af82dab82cb6c6cd77191(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4, 1, 25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1412b98f9867a93262fc89ac68140082(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d082bdf7603af82dab82cb6c6cd77191
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4, 1, 25, 38], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_261bd3021b91c14faf27cd76942437f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 1, 25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_90f54464743169aedc62f511a7cf0448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_261bd3021b91c14faf27cd76942437f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5866e07ffc06e786e8cc63a7a8559fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dffe5ee239f91530cab66dd52f24ef43
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5866e07ffc06e786e8cc63a7a8559fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dffe5ee239f91530cab66dd52f24ef43
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4fdc5d7feb41d1d34783831e2f42cb45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e01ae197afd5aa161f2954e101347f83
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4fdc5d7feb41d1d34783831e2f42cb45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e01ae197afd5aa161f2954e101347f83
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ba9981dd68ba200d455f6a39c254ed24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c23d755f2224cc7ffc2d2609055ea3
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ba9981dd68ba200d455f6a39c254ed24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c23d755f2224cc7ffc2d2609055ea3
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7bf8c384aaef90ca77b0c71455736047(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_519407c9823bdee6d6b86feab6d11c70
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7bf8c384aaef90ca77b0c71455736047(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_519407c9823bdee6d6b86feab6d11c70
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c3db7e54fc98702a6a719d370673b455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39d555b3e5c8721d2d6273764738e2a5
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c3db7e54fc98702a6a719d370673b455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39d555b3e5c8721d2d6273764738e2a5
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d347c6abc84c6222fee5e77bafcb8927(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04febfe6e5f20f763ca0126890a31c04
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d347c6abc84c6222fee5e77bafcb8927(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04febfe6e5f20f763ca0126890a31c04
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2468c61eef5edcc5d7aedb9ef8aeab54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9aa7ecb13b166316c2c5c04c93db62b
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2468c61eef5edcc5d7aedb9ef8aeab54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9aa7ecb13b166316c2c5c04c93db62b
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9cccd621cc3421138f9edabf34494595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6788cd52d4620f93775337d648c1c774
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9cccd621cc3421138f9edabf34494595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6788cd52d4620f93775337d648c1c774
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_51de3ec31ab1ea3848e4c265af096298(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 8, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2916373a1ab810481bbfa1820e31553e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51de3ec31ab1ea3848e4c265af096298
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.11497446894645691], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d987f8ca2b7f7708dd5581551b7e5071(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_555260b46f5d4a6a72a025806b8cdef9
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(10816.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_b96bf5cb1822fdef70540ddc601a781a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 152, 272], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b5d022a6be5d470cfb00546d630e42f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b96bf5cb1822fdef70540ddc601a781a
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d3d86648b8c5fef0c642aa3efce63e80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9765427904702b3cec29af184420b13e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(10816.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c536d82721744cc0af3ecbee4d7235a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c536d82721744cc0af3ecbee4d7235a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_df8207840e16cfbb4239d49254255314(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad6ecad809d350691b47c24d8a26d762
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1335b58afd0bccf325beb2e500e46bf1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 144, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_875cba435f696ba853df0f791fd88a1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1335b58afd0bccf325beb2e500e46bf1
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_875cba435f696ba853df0f791fd88a1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1335b58afd0bccf325beb2e500e46bf1
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9f3310466b56fc07e61f7e9089e0f6e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9765427904702b3cec29af184420b13e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(33856.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5bf4adf3e787052cbeb6d31049b93e7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(1.755462884902954, dtype='float32').reshape([]),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c54e5ca28f7f03fa09201306562299e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-0.44476887583732605, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_61e7931d86ec467c5b0e4630a2fa5050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0ae64e72e5aa4e75259ac44863cd3ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8032a7476c099fb04877e5b6bce55d1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ac828c055996baae99cc2e1e97ae67
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8032a7476c099fb04877e5b6bce55d1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ac828c055996baae99cc2e1e97ae67
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2794f233f52bd7c24e3e81875da463e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2794f233f52bd7c24e3e81875da463e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_593358987e9421e08775fa838144a35c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9765427904702b3cec29af184420b13e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(5776.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cdfff13bd38db149b8bec0edba751f69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9288920304028b1ae986841ae437f48b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c536d82721744cc0af3ecbee4d7235a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c536d82721744cc0af3ecbee4d7235a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_88d7001b69c98573ed16c0405ff8b851(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(5618.39306640625, dtype='float32').reshape([]),
            paddle.to_tensor(16896.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_27daf0ffba98ba7d17a7ff33d2bc134d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_27daf0ffba98ba7d17a7ff33d2bc134d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_47b0d5a6abd178147f98b010c4b624d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(19823.0703125, dtype='float32').reshape([]),
            paddle.to_tensor([-0.4938170611858368], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a63cd8b4edd389f511dd58192cb9cc5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c971ce20dee48bdfdae07a1602d817fa
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c973b04f0bbbdfafa698f83d8d7c0b5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(53.69636917114258, dtype='float32').reshape([]),
            paddle.to_tensor([-0.4938170611858368], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e390f650bcdbadca4070bdaaab94f216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f209460593495992a15bc5caa65ec1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(65536.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_029cdfebc5a2e02580b06d90a235676f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a47274817015a7349f92259db4706444
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_029cdfebc5a2e02580b06d90a235676f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a47274817015a7349f92259db4706444
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c086e20239c0eab988be062a81709fc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7045764ff1606a1c2114ef7d98ab1c2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_27345eb8c1a6e317b2bcdccdcde70b1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4096.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5866e07ffc06e786e8cc63a7a8559fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dffe5ee239f91530cab66dd52f24ef43
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5866e07ffc06e786e8cc63a7a8559fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dffe5ee239f91530cab66dd52f24ef43
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6f180beb277d4a748b88a838c3104c69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(0.5056498050689697, dtype='float32').reshape([]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7e1f1271690e70375c15c3d7e621d6eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6804, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[6804, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4f49de5cc14ef58fbd9368ca2b046dff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e1f1271690e70375c15c3d7e621d6eb
    def get_inputs(self):
        return [
            paddle.uniform([6804, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6804, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_345d2d085813649aee260a0dfb3ff397(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5376, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[5376, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a6564283e7cbf797afc052bb56be0776(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_345d2d085813649aee260a0dfb3ff397
    def get_inputs(self):
        return [
            paddle.uniform([5376, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5376, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e41c5f5b0e09e6ceeace408d23f6dc58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8daed9b9727979c396ac071efc66db1
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(16384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e1e9dccb09d1c802cba2d905c80aa8e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e1e9dccb09d1c802cba2d905c80aa8e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6332714bfb39069f6630d2611dbca3e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13cdb09e433b59dd353aef918254000
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(8464.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8032a7476c099fb04877e5b6bce55d1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ac828c055996baae99cc2e1e97ae67
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8032a7476c099fb04877e5b6bce55d1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ac828c055996baae99cc2e1e97ae67
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1663b2787ef0db1dde2e407c28723a82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8a91bff8f42093c941ca58c9512caf2
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_55cf3e66d20d0a852e4bfd79fb1c3909(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 144, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57eaa5a45d7062a71d56c1a8a5e14604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55cf3e66d20d0a852e4bfd79fb1c3909
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_57eaa5a45d7062a71d56c1a8a5e14604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55cf3e66d20d0a852e4bfd79fb1c3909
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1700a303d535bba98fa8f11e74c7d9d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13cdb09e433b59dd353aef918254000
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(400.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_17adbc3a8b5c9bf63ca2407587e84d12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(4.24906063079834, dtype='float32').reshape([]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f51dd3f6e6bac7ace6503e9254f82c95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(6184.4052734375, dtype='float32').reshape([]),
            paddle.to_tensor(18628.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ab712ac24fb77a015e977a54e7b5ade6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab712ac24fb77a015e977a54e7b5ade6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2f0b11d8e64f904bbcc54e8e0af4b4a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(29461.263671875, dtype='float32').reshape([]),
            paddle.to_tensor([-0.31651824712753296], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_47a0addcffcd25fd896a06d602271bd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c971ce20dee48bdfdae07a1602d817fa
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f969c40639d093e70d6d767c0231f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-118.61106872558594, dtype='float32').reshape([]),
            paddle.to_tensor([-0.31651824712753296], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_790b164c6511ad385f00bb714ace8273(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3546e8dc8a6842daa2ebd91dc0d08e39
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_790b164c6511ad385f00bb714ace8273(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3546e8dc8a6842daa2ebd91dc0d08e39
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d72ac10bd2f917af9d83893dec134653(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99603eb965a06cc637c6f16eebe6c271
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_048890a093f8fae44ee60cdb2d2208ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e714cc16ce2ccdf5566e9d0436e6f3de
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_048890a093f8fae44ee60cdb2d2208ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e714cc16ce2ccdf5566e9d0436e6f3de
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e1e9dccb09d1c802cba2d905c80aa8e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e1e9dccb09d1c802cba2d905c80aa8e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c6e602433e057e71185c638724d3e90f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(4989.41650390625, dtype='float32').reshape([]),
            paddle.to_tensor(15080.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4b5aa0ac912b61fec9871e5791ee64d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4b5aa0ac912b61fec9871e5791ee64d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d3f0017a969d053142d3319a86cc898e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(28475.60546875, dtype='float32').reshape([]),
            paddle.to_tensor([-0.16679665446281433], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3e7df0b0629a7ae58dd431a583a7ac33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c971ce20dee48bdfdae07a1602d817fa
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c84b89d96dd3d424b83547080c8568a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-237.65907287597656, dtype='float32').reshape([]),
            paddle.to_tensor([-0.16679665446281433], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9c091cbba25152ff2749cf59bff8fda6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2248fee1ba62d0098c3f4d5b87e40043
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9c091cbba25152ff2749cf59bff8fda6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2248fee1ba62d0098c3f4d5b87e40043
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_799fd4483319a2fa7e336cbaf1de6ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f05a1746a091fe779f485a29168fe33
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 100, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9d9cda82bcd589bfe3aac88b8b4590b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9935249d185b359b70b3eaa5cb664cdf
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9d9cda82bcd589bfe3aac88b8b4590b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9935249d185b359b70b3eaa5cb664cdf
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_27ad0336033287ca716c19f2b8765ccb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1174, 1174], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6, 1174, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d69770efd2b864912762fac7cf34a9be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27ad0336033287ca716c19f2b8765ccb
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6, 1174, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_465b67050c3c534fe14d458e3bb0ce15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0928d1a760bb9d25cd90186c07cd625e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_965b5ece88626f770040c7586420116d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6069, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f37b954fb28357c1af1ff1731830de0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_965b5ece88626f770040c7586420116d
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6069, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a00cd5201d5e3f222b210b8c7ce74030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13cdb09e433b59dd353aef918254000
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2304.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_5b496af442bd83898aeb4b722df7bb82(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 4, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3549, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6edb276fa31ded0dcef7a123d665c8f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b496af442bd83898aeb4b722df7bb82
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3549, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8ee41dd55e54ae35649780b5c5e0ca0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13cdb09e433b59dd353aef918254000
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1444.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_957e930e2de436bedb2337ad5900a802(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3024, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57965f78aadb7ce517a5d4bf7bfca437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_957e930e2de436bedb2337ad5900a802
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3024, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_371f29e24c5bcdee7bf4256a38c49f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcff2a22176319290947c71a4a6ef2fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_371f29e24c5bcdee7bf4256a38c49f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcff2a22176319290947c71a4a6ef2fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_728e433aa5805bfb4f5d20ce41bc2c8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(100.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5c2a839593a58355c519efb1891fb92f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_555260b46f5d4a6a72a025806b8cdef9
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(23104.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_c94e349a450aa3b93418d1d332cdd7ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 576, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a4a0105bed41b91d7ef16dc88832c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94e349a450aa3b93418d1d332cdd7ab
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2a4a0105bed41b91d7ef16dc88832c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94e349a450aa3b93418d1d332cdd7ab
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_369a28e5bd1118559452fb793bdf0f8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42844b690326291e4cdd8ea95897c54d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d969c58d698541fab4258ecbb749def2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eceeae894259106eef06544586420874
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(196.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0fa07117f0d3e7784b7bd3e3a8506ebc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9765427904702b3cec29af184420b13e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(16384.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_1b55e6c640098b1b5f55c96425d2b8ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 38, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 38, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_608dd93ceefdfed724b7e8e027de3c08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b55e6c640098b1b5f55c96425d2b8ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 38, 38], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_371f29e24c5bcdee7bf4256a38c49f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcff2a22176319290947c71a4a6ef2fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_371f29e24c5bcdee7bf4256a38c49f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcff2a22176319290947c71a4a6ef2fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]




if __name__ == '__main__':
    unittest.main()